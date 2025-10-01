"""State Manager for multi-step conversation workflows."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import pandas as pd

from src.utils.exceptions import (
    InvalidStateTransitionError,
    PrerequisiteNotMetError,
    SessionNotFoundError,
    DataSizeLimitError,
    SessionLimitError
)


class WorkflowType(Enum):
    """Types of workflows supported by the state manager."""
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    STATS_ANALYSIS = "stats_analysis"
    DATA_EXPLORATION = "data_exploration"


class MLTrainingState(Enum):
    """States for ML training workflow."""
    AWAITING_DATA = "awaiting_data"
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIRMING_MODEL = "confirming_model"
    TRAINING = "training"
    COMPLETE = "complete"


class MLPredictionState(Enum):
    """States for ML prediction workflow."""
    AWAITING_MODEL = "awaiting_model"
    AWAITING_DATA = "awaiting_data"
    PREDICTING = "predicting"
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


@dataclass
class StateManagerConfig:
    """Configuration for StateManager."""
    session_timeout_minutes: int = 30
    grace_period_minutes: int = 5
    max_data_size_mb: int = 100
    max_history_messages: int = 50
    cleanup_interval_seconds: int = 300
    max_concurrent_sessions: int = 1000

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
    None: {MLTrainingState.AWAITING_DATA.value},
    MLTrainingState.AWAITING_DATA.value: {MLTrainingState.SELECTING_TARGET.value},
    MLTrainingState.SELECTING_TARGET.value: {MLTrainingState.SELECTING_FEATURES.value},
    MLTrainingState.SELECTING_FEATURES.value: {MLTrainingState.CONFIRMING_MODEL.value},
    MLTrainingState.CONFIRMING_MODEL.value: {MLTrainingState.TRAINING.value},
    MLTrainingState.TRAINING.value: {MLTrainingState.COMPLETE.value},
    MLTrainingState.COMPLETE.value: set()
}

ML_PREDICTION_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    None: {MLPredictionState.AWAITING_MODEL.value},
    MLPredictionState.AWAITING_MODEL.value: {MLPredictionState.AWAITING_DATA.value},
    MLPredictionState.AWAITING_DATA.value: {MLPredictionState.PREDICTING.value},
    MLPredictionState.PREDICTING.value: {MLPredictionState.COMPLETE.value},
    MLPredictionState.COMPLETE.value: set()
}

WORKFLOW_TRANSITIONS: Dict[WorkflowType, Dict[Optional[str], Set[str]]] = {
    WorkflowType.ML_TRAINING: ML_TRAINING_TRANSITIONS,
    WorkflowType.ML_PREDICTION: ML_PREDICTION_TRANSITIONS,
}

ML_TRAINING_PREREQUISITES: Dict[str, PrerequisiteChecker] = {
    MLTrainingState.SELECTING_TARGET.value: lambda s: s.uploaded_data is not None,
    MLTrainingState.SELECTING_FEATURES.value: lambda s: 'target' in s.selections,
    MLTrainingState.CONFIRMING_MODEL.value: lambda s: 'features' in s.selections,
    MLTrainingState.TRAINING.value: lambda s: 'model_type' in s.selections,
}

ML_PREDICTION_PREREQUISITES: Dict[str, PrerequisiteChecker] = {
    MLPredictionState.AWAITING_DATA.value: lambda s: len(s.model_ids) > 0,
    MLPredictionState.PREDICTING.value: lambda s: s.uploaded_data is not None,
}

WORKFLOW_PREREQUISITES: Dict[WorkflowType, Dict[str, PrerequisiteChecker]] = {
    WorkflowType.ML_TRAINING: ML_TRAINING_PREREQUISITES,
    WorkflowType.ML_PREDICTION: ML_PREDICTION_PREREQUISITES,
}

# Prerequisite name mapping for error messages
PREREQUISITE_NAMES: Dict[Tuple[WorkflowType, str], str] = {
    (WorkflowType.ML_TRAINING, MLTrainingState.SELECTING_TARGET.value): "uploaded_data",
    (WorkflowType.ML_TRAINING, MLTrainingState.SELECTING_FEATURES.value): "target_selection",
    (WorkflowType.ML_TRAINING, MLTrainingState.CONFIRMING_MODEL.value): "feature_selection",
    (WorkflowType.ML_TRAINING, MLTrainingState.TRAINING.value): "model_type_selection",
    (WorkflowType.ML_PREDICTION, MLPredictionState.AWAITING_DATA.value): "trained_model",
    (WorkflowType.ML_PREDICTION, MLPredictionState.PREDICTING.value): "prediction_data",
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

    def __init__(self, config: Optional[StateManagerConfig] = None):
        self.config = config or StateManagerConfig()
        self._sessions: Dict[str, UserSession] = {}
        self._global_lock = asyncio.Lock()

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

    async def get_session(self, user_id: int, conversation_id: str) -> Optional[UserSession]:
        """Get existing session without creating."""
        session_key = self._get_session_key(user_id, conversation_id)

        async with self._global_lock:
            if session_key not in self._sessions:
                return None

            session = self._sessions[session_key]
            if await self._check_and_cleanup_expired(session_key, session):
                return None

            session.update_activity()
            return session

    async def update_session(self, session: UserSession) -> None:
        """Update existing session."""
        await self._update_and_save(session)

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
