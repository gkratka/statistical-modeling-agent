"""State Manager for multi-step conversation workflows."""

import asyncio
import hashlib
import hmac
import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import pandas as pd

from src.utils.exceptions import (
    InvalidStateTransitionError,
    PrerequisiteNotMetError,
    SessionNotFoundError,
    DataSizeLimitError,
    SessionLimitError
)
from src.core.state_history import StateHistory, get_fields_to_clear


class WorkflowType(Enum):
    """Types of workflows supported by the state manager."""
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    STATS_ANALYSIS = "stats_analysis"
    DATA_EXPLORATION = "data_exploration"
    SCORE_WORKFLOW = "score_workflow"  # NEW: Combined train + predict workflow
    MODELS_BROWSER = "models_browser"  # NEW: /models command - interactive model catalog
    JOIN_WORKFLOW = "join_workflow"    # NEW: /join command - dataframe join/union/concat/merge


class MLTrainingState(Enum):
    """States for ML training workflow."""
    # NEW (Phase 4): Local file path training states
    CHOOSING_DATA_SOURCE = "choosing_data_source"  # Choose: Telegram upload or local path
    AWAITING_FILE_PATH = "awaiting_file_path"      # User provides local file path
    AWAITING_PASSWORD = "awaiting_password"        # NEW: Password entry for non-whitelisted path
    CONFIRMING_SCHEMA = "confirming_schema"        # User confirms auto-detected schema

    # NEW (Phase 5): Deferred loading workflow states
    CHOOSING_LOAD_OPTION = "choosing_load_option"  # Choose: immediate load or defer
    AWAITING_SCHEMA_INPUT = "awaiting_schema_input"  # User provides manual schema

    # Existing states
    AWAITING_DATA = "awaiting_data"                # Telegram upload or post-schema confirmation
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIRMING_MODEL = "confirming_model"
    SPECIFYING_ARCHITECTURE = "specifying_architecture"  # Keras only
    COLLECTING_HYPERPARAMETERS = "collecting_hyperparameters"  # Keras only
    TRAINING = "training"

    # NEW (Model Naming Feature): Model naming workflow states
    TRAINING_COMPLETE = "training_complete"        # Training finished, showing naming options
    NAMING_MODEL = "naming_model"                  # User is entering custom name
    MODEL_NAMED = "model_named"                    # Name set, workflow complete

    COMPLETE = "complete"

    # NEW (Phase 6): Template workflow states
    SAVING_TEMPLATE = "saving_template"            # User saving template
    LOADING_TEMPLATE = "loading_template"          # User browsing templates
    CONFIRMING_TEMPLATE = "confirming_template"    # User confirming template selection
    AWAITING_TRAIN_TEMPLATE_UPLOAD = "awaiting_train_template_upload"  # Waiting for JSON file upload

    # CatBoost configuration states
    CONFIGURING_CATBOOST_ITERATIONS = "configuring_catboost_iterations"
    AWAITING_CATBOOST_ITERATIONS_INPUT = "awaiting_catboost_iterations_input"
    CONFIGURING_CATBOOST_DEPTH = "configuring_catboost_depth"
    AWAITING_CATBOOST_DEPTH_INPUT = "awaiting_catboost_depth_input"
    CONFIGURING_CATBOOST_LEARNING_RATE = "configuring_catboost_learning_rate"
    AWAITING_CATBOOST_LR_INPUT = "awaiting_catboost_lr_input"
    CONFIGURING_CATBOOST_L2_LEAF_REG = "configuring_catboost_l2_leaf_reg"
    AWAITING_CATBOOST_L2_INPUT = "awaiting_catboost_l2_input"
    CONFIGURING_CATBOOST_CATEGORICAL = "configuring_catboost_categorical"
    SELECTING_CATBOOST_CATEGORICAL = "selecting_catboost_categorical"
    CONFIRMING_CATBOOST_CONFIG = "confirming_catboost_config"


class MLPredictionState(Enum):
    """States for ML prediction workflow."""
    STARTED = "started"                                    # Initial state after /predict
    CHOOSING_DATA_SOURCE = "choosing_data_source"          # Select upload vs local path
    AWAITING_FILE_UPLOAD = "awaiting_file_upload"          # Waiting for Telegram upload
    AWAITING_FILE_PATH = "awaiting_file_path"              # Waiting for local path input
    AWAITING_PASSWORD = "awaiting_password"                # NEW: Password entry for non-whitelisted path
    CHOOSING_LOAD_OPTION = "choosing_load_option"          # Choose: immediate load or defer
    CONFIRMING_SCHEMA = "confirming_schema"                # Show schema for confirmation
    AWAITING_FEATURE_SELECTION = "awaiting_feature_selection"  # User selects features
    SELECTING_MODEL = "selecting_model"                    # Show model list
    CONFIRMING_PREDICTION_COLUMN = "confirming_prediction_column"  # Confirm column name
    READY_TO_RUN = "ready_to_run"                          # Show run/back options
    RUNNING_PREDICTION = "running_prediction"              # Executing prediction
    COMPLETE = "complete"                                  # Workflow finished

    # NEW: Local file save workflow states
    AWAITING_SAVE_PATH = "awaiting_save_path"              # User provides output directory
    AWAITING_SAVE_PASSWORD = "awaiting_save_password"      # Password entry for non-whitelisted save path
    CONFIRMING_SAVE_FILENAME = "confirming_save_filename"  # User confirms filename

    # NEW: Prediction template workflow states
    LOADING_PRED_TEMPLATE = "loading_pred_template"        # Browsing prediction templates
    CONFIRMING_PRED_TEMPLATE = "confirming_pred_template"  # Reviewing selected template
    SAVING_PRED_TEMPLATE = "saving_pred_template"          # Entering template name
    AWAITING_PRED_TEMPLATE_UPLOAD = "awaiting_pred_template_upload"  # Waiting for JSON file upload


class ScoreWorkflowState(Enum):
    """States for score workflow (combined train + predict)."""
    AWAITING_TEMPLATE = "awaiting_template"        # User submits template text
    VALIDATING_INPUTS = "validating_inputs"        # Validating paths and schemas
    CONFIRMING_EXECUTION = "confirming_execution"  # User confirms configuration
    TRAINING_MODEL = "training_model"              # Training ML model
    RUNNING_PREDICTION = "running_prediction"      # Generating predictions
    COMPLETE = "complete"                          # Workflow finished


class ModelsBrowserState(Enum):
    """States for models browser workflow (/models command)."""
    VIEWING_MODEL_LIST = "viewing_model_list"      # User browsing paginated model list
    VIEWING_MODEL_DETAILS = "viewing_model_details"  # User viewing single model details


class JoinWorkflowState(Enum):
    """States for join workflow (/join command - dataframe operations)."""
    # Step 1: Choose operation type
    CHOOSING_OPERATION = "choosing_operation"

    # Step 2: Choose number of dataframes
    CHOOSING_DATAFRAME_COUNT = "choosing_dataframe_count"

    # Steps 3-N: Collect dataframes (up to 4)
    AWAITING_DF1_SOURCE = "awaiting_df1_source"    # Upload vs Local Path
    AWAITING_DF1_PATH = "awaiting_df1_path"        # Local path input
    AWAITING_DF1_UPLOAD = "awaiting_df1_upload"    # Telegram upload

    AWAITING_DF2_SOURCE = "awaiting_df2_source"
    AWAITING_DF2_PATH = "awaiting_df2_path"
    AWAITING_DF2_UPLOAD = "awaiting_df2_upload"

    AWAITING_DF3_SOURCE = "awaiting_df3_source"
    AWAITING_DF3_PATH = "awaiting_df3_path"
    AWAITING_DF3_UPLOAD = "awaiting_df3_upload"

    AWAITING_DF4_SOURCE = "awaiting_df4_source"
    AWAITING_DF4_PATH = "awaiting_df4_path"
    AWAITING_DF4_UPLOAD = "awaiting_df4_upload"

    # Step N+1: Key column selection (for join operations only)
    CHOOSING_KEY_COLUMNS = "choosing_key_columns"

    # Step N+2: Optional filter selection (NEW)
    CHOOSING_FILTER = "choosing_filter"            # Show filter prompt with "No Filters" button
    AWAITING_FILTER_INPUT = "awaiting_filter_input"  # User typing filter expression

    # Step N+3: Output path selection
    CHOOSING_OUTPUT_PATH = "choosing_output_path"
    AWAITING_CUSTOM_OUTPUT_PATH = "awaiting_custom_output_path"

    # Step N+4: Execution
    EXECUTING_JOIN = "executing_join"
    COMPLETE = "complete"


@dataclass
class UserSession:
    """Container for user session state."""
    user_id: int
    conversation_id: str
    workflow_type: Optional[WorkflowType] = None
    current_state: Optional[str] = None
    uploaded_data: Optional[pd.DataFrame] = None
    selections: Dict[str, Any] = field(default_factory=dict)
    model_ids: List[str] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)

    # NEW (Phase 4): Local path workflow data
    data_source: Optional[str] = None  # "telegram" or "local_path"
    file_path: Optional[str] = None    # Original file path if local
    detected_schema: Optional[Dict[str, Any]] = None  # Auto-detected schema info

    # NEW (Phase 5): Deferred loading workflow data
    load_deferred: bool = False  # True if data loading is deferred until training
    manual_schema: Optional[Dict[str, Any]] = None  # User-provided schema for deferred loading

    # NEW: Back button workflow navigation
    state_history: StateHistory = field(default_factory=lambda: StateHistory(max_depth=10))
    last_back_action: Optional[float] = None  # Timestamp of last back button press (for debouncing)

    # NEW: Prediction workflow - compatible models list for button index lookup
    compatible_models: Optional[List[Dict[str, Any]]] = None  # Stores models for index-based button selection

    # NEW: i18n support - user language preference
    language: str = "en"  # User's preferred language (ISO 639-1: en, pt)
    language_detected_at: Optional[datetime] = None  # When language was detected
    language_detection_confidence: float = 0.0  # Detection confidence (0-1)

    # NEW: Password-protected path access
    dynamic_allowed_directories: List[str] = field(default_factory=list)  # Session-scoped whitelist expansion
    pending_auth_path: Optional[str] = None  # Path waiting for password authentication
    password_attempts: int = 0  # Track attempts for rate limiting

    def __post_init__(self) -> None:
        if self.user_id <= 0:
            raise ValueError("user_id must be positive")
        if not self.conversation_id:
            raise ValueError("conversation_id cannot be empty")

    @property
    def session_key(self) -> str:
        return f"{self.user_id}_{self.conversation_id}"

    def get_time_delta_minutes(self) -> float:
        """Get minutes since last activity."""
        return (datetime.now() - self.last_activity).total_seconds() / 60

    def is_expired(self, timeout_minutes: int) -> bool:
        return self.get_time_delta_minutes() > timeout_minutes

    def time_until_timeout(self, timeout_minutes: int) -> float:
        return max(0, timeout_minutes - self.get_time_delta_minutes())

    def update_activity(self) -> None:
        self.last_activity = datetime.now()

    def add_to_history(self, role: str, message: str, max_history: int = 50) -> None:
        self.history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat()
        })
        if len(self.history) > max_history:
            self.history = self.history[-max_history:]

    def get_data_size_mb(self) -> float:
        if self.uploaded_data is None:
            return 0.0
        return self.uploaded_data.memory_usage(deep=True).sum() / (1024 * 1024)

    # =========================================================================
    # Back Button Navigation Methods (NEW)
    # =========================================================================

    def save_state_snapshot(self) -> None:
        """
        Save current state to history before transition.

        Call this method BEFORE transitioning to a new state to enable
        back button navigation.
        """
        from src.core.state_history import StateSnapshot
        snapshot = StateSnapshot(self)
        self.state_history.push(snapshot)

    def restore_previous_state(self) -> bool:
        """
        Restore previous state from history and clear downstream fields.

        This implements the "not retain previous choices" requirement by:
        1. Popping the most recent snapshot from history
        2. Restoring core state (step, workflow)
        3. Restoring selections from snapshot
        4. Clearing fields set AFTER the restored state

        Returns:
            True if restoration successful, False if history is empty
        """
        previous = self.state_history.pop()
        if not previous:
            return False

        # Restore core state
        self.current_state = previous.step
        self.workflow_type = WorkflowType(previous.workflow) if previous.workflow else None

        # Restore data reference (shallow copy from snapshot)
        if previous.data_ref is not None:
            self.uploaded_data = previous.data_ref

        # Restore selections (from deep copy in snapshot)
        for key, value in previous.selections.items():
            setattr(self, key, value)

        # Restore metadata
        self.file_path = previous.file_path
        self.detected_schema = previous.detected_schema

        # Clear fields set AFTER this state (clean slate requirement)
        fields_to_clear = get_fields_to_clear(self.current_state)
        self.clear_fields(fields_to_clear)

        return True

    def clear_fields(self, field_list: List[str]) -> None:
        """
        Clear specified state fields.

        Args:
            field_list: List of field names to set to None
        """
        for field in field_list:
            if hasattr(self, field):
                setattr(self, field, None)

    def can_go_back(self) -> bool:
        """
        Check if back navigation is possible.

        Returns:
            True if history contains at least one snapshot
        """
        return self.state_history.can_go_back()

    def clear_history(self) -> None:
        """Clear all state history (e.g., on workflow restart)."""
        self.state_history.clear()


@dataclass
class StateManagerConfig:
    """Configuration for StateManager."""
    session_timeout_minutes: int = 30
    grace_period_minutes: int = 5
    max_data_size_mb: int = 100
    max_history_messages: int = 50
    cleanup_interval_seconds: int = 300
    max_concurrent_sessions: int = 1000
    sessions_dir: str = ".sessions"  # NEW: Directory for persistent sessions
    auto_save: bool = False  # NEW: Auto-save on state change

    def __post_init__(self) -> None:
        self._validate_positive("session_timeout_minutes", self.session_timeout_minutes)
        if self.grace_period_minutes < 0:
            raise ValueError("grace_period_minutes cannot be negative")
        self._validate_positive("max_data_size_mb", self.max_data_size_mb)
        self._validate_positive("max_history_messages", self.max_history_messages)
        self._validate_positive("cleanup_interval_seconds", self.cleanup_interval_seconds)
        self._validate_positive("max_concurrent_sessions", self.max_concurrent_sessions)

    def _validate_positive(self, name: str, value: int) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive")


# State Machine Configuration
PrerequisiteChecker = Callable[[UserSession], bool]

ML_TRAINING_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # Start: Choose data source (NEW) or legacy AWAITING_DATA
    None: {
        MLTrainingState.CHOOSING_DATA_SOURCE.value,  # NEW: Choose upload method
        MLTrainingState.AWAITING_DATA.value           # Legacy: Direct to upload
    },

    # NEW: Local path workflow
    MLTrainingState.CHOOSING_DATA_SOURCE.value: {
        MLTrainingState.AWAITING_FILE_PATH.value,    # User chose local path
        MLTrainingState.AWAITING_DATA.value,         # User chose Telegram upload
        MLTrainingState.LOADING_TEMPLATE.value       # User chose "Use Template"
    },
    MLTrainingState.AWAITING_FILE_PATH.value: {
        MLTrainingState.AWAITING_PASSWORD.value,     # NEW: Path not in whitelist
        MLTrainingState.CHOOSING_LOAD_OPTION.value   # Path valid, choose load strategy
    },
    MLTrainingState.AWAITING_PASSWORD.value: {
        MLTrainingState.CHOOSING_LOAD_OPTION.value,  # NEW: Password correct
        MLTrainingState.AWAITING_FILE_PATH.value      # NEW: Password incorrect, retry
    },

    # NEW: Deferred loading workflow
    MLTrainingState.CHOOSING_LOAD_OPTION.value: {
        MLTrainingState.CONFIRMING_SCHEMA.value,     # Immediate: load now, show schema
        MLTrainingState.AWAITING_SCHEMA_INPUT.value  # Defer: user provides schema
    },
    MLTrainingState.AWAITING_SCHEMA_INPUT.value: {
        MLTrainingState.SELECTING_TARGET.value,      # Schema provided, continue workflow (legacy)
        MLTrainingState.CONFIRMING_MODEL.value       # Schema complete (target+features), skip selection
    },

    MLTrainingState.CONFIRMING_SCHEMA.value: {
        MLTrainingState.SELECTING_TARGET.value,      # Schema accepted, use suggestions (legacy)
        MLTrainingState.CONFIRMING_MODEL.value,      # Schema accepted, skip to model selection (new fix)
        MLTrainingState.AWAITING_FILE_PATH.value     # Schema rejected, try different file
    },

    # Existing workflow (unchanged)
    MLTrainingState.AWAITING_DATA.value: {MLTrainingState.SELECTING_TARGET.value},
    MLTrainingState.SELECTING_TARGET.value: {MLTrainingState.SELECTING_FEATURES.value},
    MLTrainingState.SELECTING_FEATURES.value: {MLTrainingState.CONFIRMING_MODEL.value},
    MLTrainingState.CONFIRMING_MODEL.value: {
        MLTrainingState.SPECIFYING_ARCHITECTURE.value,
        MLTrainingState.TRAINING.value,
        MLTrainingState.SAVING_TEMPLATE.value
    },
    MLTrainingState.SPECIFYING_ARCHITECTURE.value: {MLTrainingState.COLLECTING_HYPERPARAMETERS.value},
    MLTrainingState.COLLECTING_HYPERPARAMETERS.value: {
        MLTrainingState.SAVING_TEMPLATE.value,       # User clicks "Save as Template"
        MLTrainingState.TRAINING.value               # User proceeds to training
    },
    MLTrainingState.TRAINING.value: {MLTrainingState.TRAINING_COMPLETE.value},

    # NEW: Model naming workflow transitions
    MLTrainingState.TRAINING_COMPLETE.value: {
        MLTrainingState.NAMING_MODEL.value,          # User clicks "Name Model"
        MLTrainingState.MODEL_NAMED.value,           # User clicks "Skip" or auto-default
        MLTrainingState.SAVING_TEMPLATE.value        # User clicks "Save as Template"
    },
    MLTrainingState.NAMING_MODEL.value: {
        MLTrainingState.MODEL_NAMED.value,           # After name provided
        MLTrainingState.TRAINING_COMPLETE.value      # Back button (optional)
    },
    MLTrainingState.MODEL_NAMED.value: {
        MLTrainingState.COMPLETE.value,
        MLTrainingState.SAVING_TEMPLATE.value  # User clicks "Save as Template" after naming
    },

    MLTrainingState.COMPLETE.value: set(),

    # NEW: Template workflow
    MLTrainingState.SAVING_TEMPLATE.value: {
        MLTrainingState.TRAINING.value,              # After saving, continue to training
        MLTrainingState.COMPLETE.value               # User cancels after saving
    },
    MLTrainingState.LOADING_TEMPLATE.value: {
        MLTrainingState.CONFIRMING_TEMPLATE.value,   # After selecting template
        MLTrainingState.AWAITING_TRAIN_TEMPLATE_UPLOAD.value  # User chooses "Upload Template"
    },
    MLTrainingState.AWAITING_TRAIN_TEMPLATE_UPLOAD.value: {
        MLTrainingState.CONFIRMING_TEMPLATE.value,   # After uploading valid template
        MLTrainingState.LOADING_TEMPLATE.value       # User cancels upload
    },
    MLTrainingState.CONFIRMING_TEMPLATE.value: {
        MLTrainingState.CHOOSING_LOAD_OPTION.value,  # Proceed with template config
        MLTrainingState.LOADING_TEMPLATE.value,      # Go back to template list
        MLTrainingState.TRAINING.value,              # Direct to training after data load (Bug #9 fix)
        MLTrainingState.COMPLETE.value               # Direct to complete when defer_loading=True
    }
}

ML_PREDICTION_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # Start: /predict command initiates workflow
    None: {MLPredictionState.STARTED.value},

    # Step 1-3: Data loading (similar to training workflow)
    MLPredictionState.STARTED.value: {
        MLPredictionState.CHOOSING_DATA_SOURCE.value,
        MLPredictionState.LOADING_PRED_TEMPLATE.value  # NEW: User chooses "Use Template"
    },
    MLPredictionState.CHOOSING_DATA_SOURCE.value: {
        MLPredictionState.AWAITING_FILE_UPLOAD.value,    # User chose Telegram upload
        MLPredictionState.AWAITING_FILE_PATH.value,      # User chose local path
        MLPredictionState.LOADING_PRED_TEMPLATE.value    # User chose "Use Template"
    },
    MLPredictionState.AWAITING_FILE_UPLOAD.value: {MLPredictionState.CONFIRMING_SCHEMA.value},
    MLPredictionState.AWAITING_FILE_PATH.value: {
        MLPredictionState.AWAITING_PASSWORD.value,       # NEW: Path not in whitelist
        MLPredictionState.CHOOSING_LOAD_OPTION.value     # Path valid, choose load strategy
    },
    MLPredictionState.AWAITING_PASSWORD.value: {
        MLPredictionState.CHOOSING_LOAD_OPTION.value,    # NEW: Password correct
        MLPredictionState.AWAITING_FILE_PATH.value        # NEW: Password incorrect, retry
    },

    # NEW: Defer loading workflow
    MLPredictionState.CHOOSING_LOAD_OPTION.value: {
        MLPredictionState.CONFIRMING_SCHEMA.value,           # Immediate: load now, show schema
        MLPredictionState.AWAITING_FEATURE_SELECTION.value   # Defer: skip to features
    },

    MLPredictionState.CONFIRMING_SCHEMA.value: {
        MLPredictionState.AWAITING_FEATURE_SELECTION.value,  # Schema accepted
        MLPredictionState.CHOOSING_DATA_SOURCE.value         # Schema rejected, go back
    },

    # Step 4-5: Feature selection
    MLPredictionState.AWAITING_FEATURE_SELECTION.value: {MLPredictionState.SELECTING_MODEL.value},

    # Step 6-7: Model selection
    MLPredictionState.SELECTING_MODEL.value: {MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value},

    # Step 8-11: Prediction column confirmation and execution
    MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value: {
        MLPredictionState.READY_TO_RUN.value,            # Column name accepted
        MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value  # User provides new name (retry)
    },
    MLPredictionState.READY_TO_RUN.value: {
        MLPredictionState.RUNNING_PREDICTION.value,      # User clicks "Run Model"
        MLPredictionState.SELECTING_MODEL.value          # User clicks "Go Back"
    },

    # Step 12-13: Execution and completion
    MLPredictionState.RUNNING_PREDICTION.value: {MLPredictionState.COMPLETE.value},

    # NEW: Local file save workflow transitions
    MLPredictionState.COMPLETE.value: {
        MLPredictionState.AWAITING_SAVE_PATH.value,      # User chooses to save locally
        MLPredictionState.SAVING_PRED_TEMPLATE.value     # NEW: User chooses "Save as Template"
    },
    MLPredictionState.AWAITING_SAVE_PATH.value: {
        MLPredictionState.AWAITING_SAVE_PASSWORD.value,   # NEW: Path not in whitelist
        MLPredictionState.CONFIRMING_SAVE_FILENAME.value  # Path validated, confirm filename
    },
    MLPredictionState.AWAITING_SAVE_PASSWORD.value: {
        MLPredictionState.CONFIRMING_SAVE_FILENAME.value,  # Password correct
        MLPredictionState.AWAITING_SAVE_PATH.value         # Password cancel, retry
    },
    MLPredictionState.CONFIRMING_SAVE_FILENAME.value: {
        MLPredictionState.COMPLETE.value                 # File saved, back to complete
    },

    # NEW: Prediction template workflow transitions
    MLPredictionState.LOADING_PRED_TEMPLATE.value: {
        MLPredictionState.CONFIRMING_PRED_TEMPLATE.value,  # User selects template
        MLPredictionState.AWAITING_PRED_TEMPLATE_UPLOAD.value,  # User chooses "Upload Template"
        MLPredictionState.STARTED.value                    # Go back to start
    },
    MLPredictionState.AWAITING_PRED_TEMPLATE_UPLOAD.value: {
        MLPredictionState.CONFIRMING_PRED_TEMPLATE.value,  # After uploading valid template
        MLPredictionState.LOADING_PRED_TEMPLATE.value      # User cancels upload
    },
    MLPredictionState.CONFIRMING_PRED_TEMPLATE.value: {
        MLPredictionState.READY_TO_RUN.value,              # After loading data, ready to run
        MLPredictionState.LOADING_PRED_TEMPLATE.value      # Go back to template list
    },
    MLPredictionState.SAVING_PRED_TEMPLATE.value: {
        MLPredictionState.COMPLETE.value                   # After save or cancel
    }
}

SCORE_WORKFLOW_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # Start: /score command initiates workflow
    None: {ScoreWorkflowState.AWAITING_TEMPLATE.value},

    # Step 1: Template submission and validation
    ScoreWorkflowState.AWAITING_TEMPLATE.value: {
        ScoreWorkflowState.VALIDATING_INPUTS.value,      # Template submitted
        ScoreWorkflowState.CONFIRMING_EXECUTION.value    # Fast path if validation passes
    },

    # Step 2: Validation (may be skipped with fast path)
    ScoreWorkflowState.VALIDATING_INPUTS.value: {
        ScoreWorkflowState.CONFIRMING_EXECUTION.value,   # Validation passed
        ScoreWorkflowState.AWAITING_TEMPLATE.value       # Validation failed, retry
    },

    # Step 3: User confirmation
    ScoreWorkflowState.CONFIRMING_EXECUTION.value: {
        ScoreWorkflowState.TRAINING_MODEL.value,         # User confirmed
        ScoreWorkflowState.AWAITING_TEMPLATE.value       # User cancelled, restart
    },

    # Step 4: Training
    ScoreWorkflowState.TRAINING_MODEL.value: {
        ScoreWorkflowState.RUNNING_PREDICTION.value      # Training complete, start prediction
    },

    # Step 5: Prediction
    ScoreWorkflowState.RUNNING_PREDICTION.value: {
        ScoreWorkflowState.COMPLETE.value                # Prediction complete
    },

    # Step 6: Complete (terminal state)
    ScoreWorkflowState.COMPLETE.value: set()
}

MODELS_BROWSER_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # Start: /models command initiates workflow
    None: {ModelsBrowserState.VIEWING_MODEL_LIST.value},

    # Step 1: Viewing model list (paginated)
    ModelsBrowserState.VIEWING_MODEL_LIST.value: {
        ModelsBrowserState.VIEWING_MODEL_DETAILS.value  # User selects a model
    },

    # Step 2: Viewing model details
    ModelsBrowserState.VIEWING_MODEL_DETAILS.value: {
        ModelsBrowserState.VIEWING_MODEL_LIST.value  # Back button to list
    }
}

JOIN_WORKFLOW_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    # Start: /join command initiates workflow
    None: {JoinWorkflowState.CHOOSING_OPERATION.value},

    # Step 1: Choose operation type
    JoinWorkflowState.CHOOSING_OPERATION.value: {
        JoinWorkflowState.CHOOSING_DATAFRAME_COUNT.value  # After selecting operation
    },

    # Step 2: Choose dataframe count
    JoinWorkflowState.CHOOSING_DATAFRAME_COUNT.value: {
        JoinWorkflowState.AWAITING_DF1_SOURCE.value  # Start collecting dataframes
    },

    # Dataframe 1 collection
    JoinWorkflowState.AWAITING_DF1_SOURCE.value: {
        JoinWorkflowState.AWAITING_DF1_PATH.value,     # User chose local path
        JoinWorkflowState.AWAITING_DF1_UPLOAD.value    # User chose upload
    },
    JoinWorkflowState.AWAITING_DF1_PATH.value: {
        JoinWorkflowState.AWAITING_DF2_SOURCE.value    # Path validated, next dataframe
    },
    JoinWorkflowState.AWAITING_DF1_UPLOAD.value: {
        JoinWorkflowState.AWAITING_DF2_SOURCE.value    # Upload received, next dataframe
    },

    # Dataframe 2 collection
    JoinWorkflowState.AWAITING_DF2_SOURCE.value: {
        JoinWorkflowState.AWAITING_DF2_PATH.value,
        JoinWorkflowState.AWAITING_DF2_UPLOAD.value
    },
    JoinWorkflowState.AWAITING_DF2_PATH.value: {
        JoinWorkflowState.AWAITING_DF3_SOURCE.value,   # If count > 2
        JoinWorkflowState.CHOOSING_KEY_COLUMNS.value,  # If count == 2 and join op
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value   # If count == 2 and union/concat
    },
    JoinWorkflowState.AWAITING_DF2_UPLOAD.value: {
        JoinWorkflowState.AWAITING_DF3_SOURCE.value,
        JoinWorkflowState.CHOOSING_KEY_COLUMNS.value,
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value
    },

    # Dataframe 3 collection (optional)
    JoinWorkflowState.AWAITING_DF3_SOURCE.value: {
        JoinWorkflowState.AWAITING_DF3_PATH.value,
        JoinWorkflowState.AWAITING_DF3_UPLOAD.value
    },
    JoinWorkflowState.AWAITING_DF3_PATH.value: {
        JoinWorkflowState.AWAITING_DF4_SOURCE.value,   # If count > 3
        JoinWorkflowState.CHOOSING_KEY_COLUMNS.value,  # If count == 3 and join op
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value   # If count == 3 and union/concat
    },
    JoinWorkflowState.AWAITING_DF3_UPLOAD.value: {
        JoinWorkflowState.AWAITING_DF4_SOURCE.value,
        JoinWorkflowState.CHOOSING_KEY_COLUMNS.value,
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value
    },

    # Dataframe 4 collection (optional)
    JoinWorkflowState.AWAITING_DF4_SOURCE.value: {
        JoinWorkflowState.AWAITING_DF4_PATH.value,
        JoinWorkflowState.AWAITING_DF4_UPLOAD.value
    },
    JoinWorkflowState.AWAITING_DF4_PATH.value: {
        JoinWorkflowState.CHOOSING_KEY_COLUMNS.value,  # If join op
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value   # If union/concat
    },
    JoinWorkflowState.AWAITING_DF4_UPLOAD.value: {
        JoinWorkflowState.CHOOSING_KEY_COLUMNS.value,
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value
    },

    # Key column selection (for join operations)
    JoinWorkflowState.CHOOSING_KEY_COLUMNS.value: {
        JoinWorkflowState.CHOOSING_OUTPUT_PATH.value   # After key columns selected
    },

    # Output path selection
    JoinWorkflowState.CHOOSING_OUTPUT_PATH.value: {
        JoinWorkflowState.AWAITING_CUSTOM_OUTPUT_PATH.value,  # User wants custom path
        JoinWorkflowState.EXECUTING_JOIN.value         # User accepts default
    },
    JoinWorkflowState.AWAITING_CUSTOM_OUTPUT_PATH.value: {
        JoinWorkflowState.EXECUTING_JOIN.value         # Custom path provided
    },

    # Execution and completion
    JoinWorkflowState.EXECUTING_JOIN.value: {
        JoinWorkflowState.COMPLETE.value               # Job finished
    },
    JoinWorkflowState.COMPLETE.value: set()            # Terminal state
}

WORKFLOW_TRANSITIONS: Dict[WorkflowType, Dict[Optional[str], Set[str]]] = {
    WorkflowType.ML_TRAINING: ML_TRAINING_TRANSITIONS,
    WorkflowType.ML_PREDICTION: ML_PREDICTION_TRANSITIONS,
    WorkflowType.SCORE_WORKFLOW: SCORE_WORKFLOW_TRANSITIONS,
    WorkflowType.MODELS_BROWSER: MODELS_BROWSER_TRANSITIONS,
    WorkflowType.JOIN_WORKFLOW: JOIN_WORKFLOW_TRANSITIONS,
}

ML_TRAINING_PREREQUISITES: Dict[str, PrerequisiteChecker] = {
    MLTrainingState.SELECTING_TARGET.value: lambda s: s.uploaded_data is not None,
    MLTrainingState.SELECTING_FEATURES.value: lambda s: 'target_column' in s.selections or 'target' in s.selections,
    MLTrainingState.CONFIRMING_MODEL.value: lambda s: 'feature_columns' in s.selections or 'features' in s.selections,
    MLTrainingState.SPECIFYING_ARCHITECTURE.value: lambda s: 'model_type' in s.selections,
    MLTrainingState.COLLECTING_HYPERPARAMETERS.value: lambda s: 'architecture' in s.selections,
    MLTrainingState.TRAINING.value: lambda s: 'model_type' in s.selections,

    # NEW: Model naming workflow prerequisites
    MLTrainingState.NAMING_MODEL.value: lambda s: 'pending_model_id' in s.selections,
    MLTrainingState.MODEL_NAMED.value: lambda s: 'pending_model_id' in s.selections,
}

ML_PREDICTION_PREREQUISITES: Dict[str, PrerequisiteChecker] = {
    # Data must be loaded before confirming schema
    MLPredictionState.CONFIRMING_SCHEMA.value: lambda s: s.uploaded_data is not None or s.file_path is not None,

    # Features must be selected before model selection
    MLPredictionState.SELECTING_MODEL.value: lambda s: 'selected_features' in s.selections and s.selections['selected_features'],

    # Model must be selected before column confirmation
    MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value: lambda s: 'selected_model_id' in s.selections,

    # Column name must be confirmed before ready to run
    MLPredictionState.READY_TO_RUN.value: lambda s: 'prediction_column_name' in s.selections,

    # All data and model must be ready before running prediction
    # DEFER LOADING FIX: Allow None data if defer loading is enabled (data loads during execution)
    MLPredictionState.RUNNING_PREDICTION.value: lambda s: (
        (s.uploaded_data is not None or getattr(s, 'load_deferred', False)) and
        'selected_model_id' in s.selections and
        'selected_features' in s.selections and
        'prediction_column_name' in s.selections
    ),
}

SCORE_WORKFLOW_PREREQUISITES: Dict[str, PrerequisiteChecker] = {
    # Configuration must exist before confirmation
    ScoreWorkflowState.CONFIRMING_EXECUTION.value: lambda s: 'score_config' in s.selections,

    # Configuration must exist before training
    ScoreWorkflowState.TRAINING_MODEL.value: lambda s: 'score_config' in s.selections,

    # Configuration must exist before prediction
    ScoreWorkflowState.RUNNING_PREDICTION.value: lambda s: 'score_config' in s.selections,
}

WORKFLOW_PREREQUISITES: Dict[WorkflowType, Dict[str, PrerequisiteChecker]] = {
    WorkflowType.ML_TRAINING: ML_TRAINING_PREREQUISITES,
    WorkflowType.ML_PREDICTION: ML_PREDICTION_PREREQUISITES,
    WorkflowType.SCORE_WORKFLOW: SCORE_WORKFLOW_PREREQUISITES,
}

# Prerequisite name mapping for error messages
PREREQUISITE_NAMES: Dict[Tuple[WorkflowType, str], str] = {
    # ML Training prerequisites
    (WorkflowType.ML_TRAINING, MLTrainingState.SELECTING_TARGET.value): "uploaded_data",
    (WorkflowType.ML_TRAINING, MLTrainingState.SELECTING_FEATURES.value): "target_selection",
    (WorkflowType.ML_TRAINING, MLTrainingState.CONFIRMING_MODEL.value): "feature_selection",
    (WorkflowType.ML_TRAINING, MLTrainingState.TRAINING.value): "model_type_selection",

    # ML Prediction prerequisites
    (WorkflowType.ML_PREDICTION, MLPredictionState.CONFIRMING_SCHEMA.value): "prediction_data",
    (WorkflowType.ML_PREDICTION, MLPredictionState.SELECTING_MODEL.value): "feature_selection",
    (WorkflowType.ML_PREDICTION, MLPredictionState.CONFIRMING_PREDICTION_COLUMN.value): "model_selection",
    (WorkflowType.ML_PREDICTION, MLPredictionState.READY_TO_RUN.value): "prediction_column_name",
    (WorkflowType.ML_PREDICTION, MLPredictionState.RUNNING_PREDICTION.value): "all_prediction_requirements",

    # Score Workflow prerequisites
    (WorkflowType.SCORE_WORKFLOW, ScoreWorkflowState.CONFIRMING_EXECUTION.value): "score_configuration",
    (WorkflowType.SCORE_WORKFLOW, ScoreWorkflowState.TRAINING_MODEL.value): "score_configuration",
    (WorkflowType.SCORE_WORKFLOW, ScoreWorkflowState.RUNNING_PREDICTION.value): "score_configuration",
}


class StateMachine:
    """State machine for workflow management."""

    @staticmethod
    def check_prerequisites(
        workflow_type: WorkflowType,
        state: str,
        session: UserSession
    ) -> Tuple[bool, List[str]]:
        """Check if prerequisites are met for entering a state."""
        if workflow_type not in WORKFLOW_PREREQUISITES:
            return True, []

        prerequisites = WORKFLOW_PREREQUISITES[workflow_type]
        if state not in prerequisites:
            return True, []

        if prerequisites[state](session):
            return True, []

        # Get prerequisite name from mapping
        prereq_name = PREREQUISITE_NAMES.get((workflow_type, state), "unknown")
        return False, [prereq_name]

    @staticmethod
    def validate_transition(
        workflow_type: WorkflowType,
        current_state: Optional[str],
        new_state: str,
        session: UserSession
    ) -> Tuple[bool, Optional[str], List[str]]:
        """Validate state transition. Returns (success, error_message, missing_prerequisites)."""
        # Check if transition is valid
        if workflow_type not in WORKFLOW_TRANSITIONS:
            return False, f"Unknown workflow type: {workflow_type.value}", []

        transitions = WORKFLOW_TRANSITIONS[workflow_type]
        if current_state not in transitions:
            return False, f"Invalid current state: {current_state}", []

        if new_state not in transitions[current_state]:
            return False, f"Invalid transition from '{current_state}' to '{new_state}'", []

        # Check prerequisites
        prereqs_met, missing = StateMachine.check_prerequisites(workflow_type, new_state, session)
        if not prereqs_met:
            return False, f"Prerequisites not met for state '{new_state}'", missing

        return True, None, []

    @staticmethod
    def get_valid_next_states(workflow_type: WorkflowType, current_state: Optional[str]) -> Set[str]:
        """Get set of valid next states from current state."""
        if workflow_type not in WORKFLOW_TRANSITIONS:
            return set()
        return WORKFLOW_TRANSITIONS[workflow_type].get(current_state, set())


class StateManager:
    """Manage user sessions and workflow state."""

    def __init__(
        self,
        config: Optional[StateManagerConfig] = None,
        sessions_dir: Optional[str] = None,
        auto_save: Optional[bool] = None
    ):
        self.config = config or StateManagerConfig()

        # Override config with constructor params if provided
        if sessions_dir is not None:
            self.config.sessions_dir = sessions_dir
        if auto_save is not None:
            self.config.auto_save = auto_save

        self._sessions: Dict[str, UserSession] = {}
        self._global_lock = asyncio.Lock()

        # Create sessions directory if it doesn't exist
        self._sessions_path = Path(self.config.sessions_dir)
        self._sessions_path.mkdir(parents=True, exist_ok=True)

    def _get_session_key(self, user_id: int, conversation_id: str) -> str:
        """Generate session key."""
        return f"{user_id}_{conversation_id}"

    async def _check_and_cleanup_expired(self, session_key: str, session: UserSession) -> bool:
        """Check if session is expired and cleanup if needed. Returns True if expired."""
        total_timeout = self.config.session_timeout_minutes + self.config.grace_period_minutes
        if session.is_expired(total_timeout):
            del self._sessions[session_key]
            return True
        return False

    async def _update_and_save(self, session: UserSession) -> None:
        """Update activity and save session."""
        session.update_activity()
        async with self._global_lock:
            if session.session_key not in self._sessions:
                raise SessionNotFoundError(f"Session {session.session_key} not found")
            self._sessions[session.session_key] = session

    async def get_or_create_session(self, user_id: int, conversation_id: str) -> UserSession:
        """Get existing session or create new one."""
        session_key = self._get_session_key(user_id, conversation_id)

        async with self._global_lock:
            if session_key in self._sessions:
                session = self._sessions[session_key]
                if await self._check_and_cleanup_expired(session_key, session):
                    pass  # Session was expired and cleaned up, create new one
                else:
                    session.update_activity()
                    return session

            if len(self._sessions) >= self.config.max_concurrent_sessions:
                raise SessionLimitError(
                    f"Maximum concurrent sessions ({self.config.max_concurrent_sessions}) reached",
                    current_count=len(self._sessions),
                    limit=self.config.max_concurrent_sessions
                )

            session = UserSession(user_id=user_id, conversation_id=conversation_id)
            self._sessions[session_key] = session
            return session

    async def get_session(
        self,
        user_id: int,
        conversation_id: str,
        auto_load: bool = False
    ) -> Optional[UserSession]:
        """Get existing session without creating. If auto_load=True, tries to load from disk."""
        session_key = self._get_session_key(user_id, conversation_id)

        async with self._global_lock:
            if session_key not in self._sessions:
                # If not in memory and auto_load enabled, try loading from disk
                if auto_load:
                    # Release lock temporarily to call load_session_from_disk
                    pass  # Will load after lock is released
                else:
                    return None

            else:
                # Session in memory - check expiry
                session = self._sessions[session_key]
                if await self._check_and_cleanup_expired(session_key, session):
                    return None

                session.update_activity()
                return session

        # If auto_load and not in memory, try loading from disk (outside lock)
        if auto_load:
            loaded_session = await self.load_session_from_disk(user_id)
            if loaded_session:
                loaded_session.update_activity()
            return loaded_session

        return None

    async def update_session(self, session: UserSession) -> None:
        """Update existing session. If auto_save enabled, also saves to disk."""
        await self._update_and_save(session)

        # Auto-save to disk if enabled
        if self.config.auto_save:
            try:
                await self.save_session_to_disk(session.user_id)
            except Exception:
                # Don't fail the update if disk save fails
                pass

    async def delete_session(self, user_id: int, conversation_id: str) -> None:
        """Delete session."""
        session_key = self._get_session_key(user_id, conversation_id)
        async with self._global_lock:
            self._sessions.pop(session_key, None)

    async def start_workflow(self, session: UserSession, workflow_type: WorkflowType) -> None:
        """Start a new workflow for session."""
        if session.workflow_type is not None:
            raise InvalidStateTransitionError(
                f"Cannot start {workflow_type.value}: workflow {session.workflow_type.value} already active"
            )

        initial_states = StateMachine.get_valid_next_states(workflow_type, None)
        if not initial_states:
            raise InvalidStateTransitionError(f"No initial state defined for workflow {workflow_type.value}")

        session.workflow_type = workflow_type
        session.current_state = list(initial_states)[0]
        await self._update_and_save(session)

    async def transition_state(self, session: UserSession, new_state: str) -> Tuple[bool, Optional[str], List[str]]:
        """Transition session to new state. Returns (success, error_message, missing_prerequisites)."""
        if session.workflow_type is None:
            raise InvalidStateTransitionError("Cannot transition state: no workflow active")

        success, error_msg, missing = StateMachine.validate_transition(
            session.workflow_type, session.current_state, new_state, session
        )

        if success:
            session.current_state = new_state
            await self._update_and_save(session)

        return success, error_msg, missing

    async def cancel_workflow(self, session: UserSession) -> None:
        """Cancel active workflow."""
        session.workflow_type = None
        session.current_state = None
        session.selections.clear()
        await self._update_and_save(session)

    async def store_data(self, session: UserSession, data: pd.DataFrame) -> None:
        """Store DataFrame in session."""
        data_size_mb = data.memory_usage(deep=True).sum() / (1024 * 1024)
        if data_size_mb > self.config.max_data_size_mb:
            raise DataSizeLimitError(
                f"Data size ({data_size_mb:.2f} MB) exceeds limit ({self.config.max_data_size_mb} MB)",
                actual_size_mb=data_size_mb,
                limit_mb=self.config.max_data_size_mb
            )

        session.uploaded_data = data
        await self._update_and_save(session)

    async def get_data(self, session: UserSession) -> Optional[pd.DataFrame]:
        """Get DataFrame from session."""
        return session.uploaded_data

    async def add_to_history(self, session: UserSession, role: str, message: str) -> None:
        """Add message to conversation history."""
        session.add_to_history(role=role, message=message, max_history=self.config.max_history_messages)
        await self._update_and_save(session)

    async def get_history(self, session: UserSession, n_messages: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history."""
        return session.history[-n_messages:] if session.history else []

    async def cleanup_expired_sessions(self) -> int:
        """Remove expired sessions."""
        total_timeout = self.config.session_timeout_minutes + self.config.grace_period_minutes
        expired_keys = []

        async with self._global_lock:
            for key, session in self._sessions.items():
                if session.is_expired(total_timeout):
                    expired_keys.append(key)

            for key in expired_keys:
                del self._sessions[key]

        return len(expired_keys)

    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        async with self._global_lock:
            return len(self._sessions)

    async def get_session_timeout_warning(self, session: UserSession) -> Optional[str]:
        """Get timeout warning message if approaching timeout."""
        time_left = session.time_until_timeout(self.config.session_timeout_minutes)
        threshold = self.config.session_timeout_minutes * 0.2

        if 0 < time_left < threshold:
            return f"⚠️ Your session will expire in {int(time_left)} minutes due to inactivity"
        return None

    # =========================================================================
    # Session Persistence Methods (Phase 3: Workflow Fix Plan)
    # =========================================================================

    def _get_session_file_path(self, user_id: int) -> Path:
        """Get path to session file for user."""
        return self._sessions_path / f"user_{user_id}.json"

    def _get_signing_key(self) -> bytes:
        """Get the session signing key from environment.

        Returns:
            bytes: The signing key

        Raises:
            ValueError: If SESSION_SIGNING_KEY environment variable is not set
        """
        key_hex = os.getenv('SESSION_SIGNING_KEY')
        if key_hex is None:
            # Return None to allow unsigned sessions in dev environments
            return None
        return bytes.fromhex(key_hex)

    def _compute_session_signature(self, data: Dict[str, Any]) -> str:
        """Compute HMAC-SHA256 signature for session data.

        Args:
            data: Session data dictionary (without signature)

        Returns:
            str: Hexadecimal signature string
        """
        key = self._get_signing_key()
        if key is None:
            return ""

        # Create deterministic JSON representation for signing
        json_str = json.dumps(data, sort_keys=True)
        signature = hmac.new(key, json_str.encode('utf-8'), hashlib.sha256)
        return signature.hexdigest()

    def _verify_session_signature(self, data: Dict[str, Any]) -> bool:
        """Verify HMAC-SHA256 signature of session data.

        Args:
            data: Session data dictionary (with signature)

        Returns:
            bool: True if signature is valid, False otherwise
        """
        key = self._get_signing_key()

        # If no signing key configured, accept unsigned sessions (dev mode)
        if key is None:
            return True

        # If signing key is set, require valid signature
        if 'signature' not in data:
            return False

        stored_signature = data['signature']

        # Create copy without signature for verification
        data_copy = {k: v for k, v in data.items() if k != 'signature'}

        # Compute expected signature
        json_str = json.dumps(data_copy, sort_keys=True)
        expected_signature = hmac.new(key, json_str.encode('utf-8'), hashlib.sha256).hexdigest()

        # Use constant-time comparison to prevent timing attacks
        return hmac.compare_digest(stored_signature, expected_signature)

    def _session_to_dict(self, session: UserSession) -> Dict[str, Any]:
        """Convert session to JSON-serializable dict."""
        data = {
            "user_id": session.user_id,
            "conversation_id": session.conversation_id,
            "workflow_type": session.workflow_type.value if session.workflow_type else None,
            "current_state": session.current_state,
            "selections": session.selections,
            "model_ids": session.model_ids,
            "history": session.history,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "data_source": session.data_source,
            "file_path": session.file_path,
            "detected_schema": session.detected_schema,
            "load_deferred": session.load_deferred,
            "manual_schema": session.manual_schema,
            "state_history": session.state_history.to_dict(),  # NEW: Serialize state history
            "last_back_action": session.last_back_action,  # NEW: Serialize debounce timestamp
            "pending_auth_path": session.pending_auth_path,  # FIX: Persist for password workflow
        }
        # Note: uploaded_data (DataFrame) is NOT persisted due to size
        # NOTE: Password fields NOT persisted (security):
        #   - dynamic_allowed_directories (session-scoped only)
        #   - password_attempts (rate limiting reset on reload)

        # Add HMAC-SHA256 signature if signing key is configured
        signature = self._compute_session_signature(data)
        if signature:
            data['signature'] = signature

        return data

    def _dict_to_session(self, data: Dict[str, Any]) -> UserSession:
        """Reconstruct session from dict."""
        session = UserSession(
            user_id=data["user_id"],
            conversation_id=data["conversation_id"]
        )
        session.workflow_type = WorkflowType(data["workflow_type"]) if data["workflow_type"] else None
        session.current_state = data["current_state"]
        session.selections = data["selections"]
        session.model_ids = data["model_ids"]
        session.history = data["history"]
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.last_activity = datetime.fromisoformat(data["last_activity"])
        session.data_source = data.get("data_source")
        session.file_path = data.get("file_path")
        session.detected_schema = data.get("detected_schema")
        session.load_deferred = data.get("load_deferred", False)
        session.manual_schema = data.get("manual_schema")

        # NEW: Deserialize state history
        if "state_history" in data:
            session.state_history = StateHistory.from_dict(data["state_history"])
        session.last_back_action = data.get("last_back_action")
        session.pending_auth_path = data.get("pending_auth_path")  # FIX: Restore for password workflow

        # uploaded_data remains None - must be reloaded if needed
        return session

    async def save_session_to_disk(self, user_id: int) -> None:
        """Save session to disk for persistence across restarts."""
        session_key = self._get_session_key(user_id, "*")  # Get any conversation for this user

        # Find session for this user
        async with self._global_lock:
            user_session = None
            for key, session in self._sessions.items():
                if session.user_id == user_id:
                    user_session = session
                    break

            if user_session is None:
                raise SessionNotFoundError(f"No session found for user {user_id}")

            # Convert to dict and save
            session_data = self._session_to_dict(user_session)

        # Write atomically using temporary file
        session_file = self._get_session_file_path(user_id)
        temp_file = session_file.with_suffix('.tmp')

        try:
            temp_file.write_text(json.dumps(session_data, indent=2))
            temp_file.replace(session_file)  # Atomic rename
        except Exception as e:
            if temp_file.exists():
                temp_file.unlink()
            raise

    async def load_session_from_disk(self, user_id: int) -> Optional[UserSession]:
        """Load session from disk. Returns None if no saved session exists.

        Security: If SESSION_SIGNING_KEY is set, verifies HMAC-SHA256 signature
        before loading. Tampered or unsigned sessions are rejected and deleted.
        """
        session_file = self._get_session_file_path(user_id)

        if not session_file.exists():
            return None

        try:
            session_data = json.loads(session_file.read_text())

            # Verify signature before loading (if signing is enabled)
            if not self._verify_session_signature(session_data):
                # Tampered or unsigned session - reject and delete
                session_file.unlink()
                return None

            # Remove signature from data before deserialization
            session_data.pop('signature', None)

            session = self._dict_to_session(session_data)

            # Add to memory cache
            async with self._global_lock:
                self._sessions[session.session_key] = session

            return session

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            # Corrupted session file - delete it
            session_file.unlink()
            return None

    async def complete_workflow(self, user_id: int) -> None:
        """Complete workflow and clean up session."""
        # Find and remove session from memory
        async with self._global_lock:
            user_session_key = None
            for key, session in self._sessions.items():
                if session.user_id == user_id:
                    user_session_key = key
                    break

            if user_session_key:
                del self._sessions[user_session_key]

        # Remove session file from disk
        session_file = self._get_session_file_path(user_id)
        if session_file.exists():
            session_file.unlink()

    # =========================================================================
    # Password-Protected Path Access Methods (Phase 2: Password Implementation)
    # =========================================================================

    def add_dynamic_directory(self, session: UserSession, directory: str) -> None:
        """Add directory to session-scoped whitelist.

        This allows temporary expansion of allowed directories after
        password authentication, without modifying the global config.

        Args:
            session: User session to modify
            directory: Directory path to add

        Security Notes:
            - Directory is added ONLY to this specific session
            - NOT persisted to disk (session-scoped only)
            - Cleared when workflow completes
        """
        if directory not in session.dynamic_allowed_directories:
            session.dynamic_allowed_directories.append(directory)
            # Log using auth_logger from password_validator module
            from src.utils.password_validator import auth_logger
            auth_logger.info(
                f"Dynamic whitelist expanded: user={session.user_id}, "
                f"dir={directory}, session={session.session_key}"
            )

    def get_effective_allowed_directories(
        self,
        session: UserSession,
        base_allowed: List[str]
    ) -> List[str]:
        """Get combined whitelist (base + dynamic).

        Args:
            session: User session
            base_allowed: Base allowed directories from config

        Returns:
            Combined list of allowed directories (duplicates removed)
        """
        return list(set(base_allowed + session.dynamic_allowed_directories))
