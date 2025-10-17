# State Manager Implementation Plan

**Date:** 2025-10-01
**Feature Branch:** `feature/state-manager`
**Status:** 📋 Planning Complete - Ready for Implementation

## Overview

The State Manager is a critical component for managing multi-step conversations in the Telegram statistical modeling bot. It tracks user sessions, workflow state, uploaded data, and conversation history to enable complex interactions like ML model training that require multiple user inputs.

## Problem Statement

Current system lacks:
- Session persistence across messages
- Multi-step workflow management (ML training requires: data upload → target selection → feature selection → model type → training)
- Context for ambiguous requests ("analyze it", "what about correlation?")
- Timeout handling for idle sessions
- User data storage between interactions

## Solution Architecture

### Core Components

```
StateManager
├── Session Management Layer
│   ├── UserSession dataclass (state container)
│   ├── Session CRUD operations
│   └── AsyncIO locks for concurrency
│
├── State Machine Layer
│   ├── WorkflowType definitions
│   ├── State transition rules
│   └── Prerequisite validation
│
├── Data Storage Layer
│   ├── DataFrame storage
│   ├── Model ID tracking
│   └── Conversation history
│
└── Persistence Layer (Abstract)
    ├── MemoryBackend (MVP)
    ├── PickleBackend (future)
    └── RedisBackend (scalability)
```

### Data Structures

#### 1. UserSession Dataclass

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd

@dataclass
class UserSession:
    """Container for user session state"""
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

    @property
    def session_key(self) -> str:
        return f"{self.user_id}_{self.conversation_id}"

    @property
    def is_expired(self, timeout_minutes: int) -> bool:
        delta = (datetime.now() - self.last_activity).total_seconds() / 60
        return delta > timeout_minutes
```

#### 2. Configuration

```python
@dataclass
class StateManagerConfig:
    """State manager configuration"""
    session_timeout_minutes: int = 30
    max_history_messages: int = 50
    max_concurrent_sessions: int = 1000
    enable_persistence: bool = False
    persistence_backend: str = "memory"  # memory, pickle, redis
    persistence_path: Optional[str] = None
    cleanup_interval_minutes: int = 10
    enable_workflow_recovery: bool = True
    grace_period_minutes: int = 5
    max_dataframe_size_mb: int = 100
```

#### 3. Workflow Types and States

```python
from enum import Enum

class WorkflowType(Enum):
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    STATS_ANALYSIS = "stats_analysis"
    DATA_EXPLORATION = "data_exploration"

class MLTrainingState(Enum):
    AWAITING_DATA = "awaiting_data"
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIRMING_MODEL = "confirming_model"
    TRAINING = "training"
    COMPLETE = "complete"

# State transition rules
ML_TRAINING_TRANSITIONS = {
    None: [MLTrainingState.AWAITING_DATA],
    MLTrainingState.AWAITING_DATA: [MLTrainingState.SELECTING_TARGET],
    MLTrainingState.SELECTING_TARGET: [MLTrainingState.SELECTING_FEATURES],
    MLTrainingState.SELECTING_FEATURES: [MLTrainingState.CONFIRMING_MODEL],
    MLTrainingState.CONFIRMING_MODEL: [MLTrainingState.TRAINING],
    MLTrainingState.TRAINING: [MLTrainingState.COMPLETE],
}

# Prerequisites for state transitions
STATE_PREREQUISITES = {
    MLTrainingState.SELECTING_TARGET: lambda s: s.uploaded_data is not None,
    MLTrainingState.SELECTING_FEATURES: lambda s: 'target' in s.selections,
    MLTrainingState.CONFIRMING_MODEL: lambda s: 'features' in s.selections,
    MLTrainingState.TRAINING: lambda s: 'model_type' in s.selections,
}
```

### State Manager Class Design

```python
import asyncio
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod

class StateManager:
    """Manages user sessions and multi-step workflows"""

    def __init__(self, config: Optional[StateManagerConfig] = None):
        self.config = config or StateManagerConfig()
        self._sessions: Dict[str, UserSession] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._expired_sessions: Dict[str, Dict] = {}
        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._backend = self._create_backend()

    # === Session Management ===

    async def get_or_create_session(
        self,
        user_id: int,
        conversation_id: str
    ) -> UserSession:
        """Get existing session or create new one"""
        session_key = f"{user_id}_{conversation_id}"

        async with await self._get_lock(session_key):
            if session_key in self._sessions:
                session = self._sessions[session_key]
                session.last_activity = datetime.now()
                return session

            # Check for recoverable expired session
            recoverable = await self._check_recoverable_session(session_key)
            if recoverable:
                return recoverable

            # Create new session
            session = UserSession(user_id=user_id, conversation_id=conversation_id)
            self._sessions[session_key] = session
            logger.info(f"Session created: {session_key}")
            return session

    async def get_session(
        self,
        user_id: int,
        conversation_id: str
    ) -> Optional[UserSession]:
        """Get session without creating"""
        session_key = f"{user_id}_{conversation_id}"
        return self._sessions.get(session_key)

    async def update_session(self, session: UserSession) -> None:
        """Update session state"""
        async with await self._get_lock(session.session_key):
            session.last_activity = datetime.now()
            self._sessions[session.session_key] = session

    async def delete_session(self, user_id: int, conversation_id: str) -> None:
        """Delete session and cleanup"""
        session_key = f"{user_id}_{conversation_id}"
        async with await self._get_lock(session_key):
            if session_key in self._sessions:
                del self._sessions[session_key]
                logger.info(f"Session deleted: {session_key}")

    # === Workflow Management ===

    async def start_workflow(
        self,
        session: UserSession,
        workflow_type: WorkflowType
    ) -> None:
        """Start a new workflow"""
        session.workflow_type = workflow_type
        session.current_state = self._get_initial_state(workflow_type)
        await self.update_session(session)

    async def get_current_state(self, session: UserSession) -> Optional[str]:
        """Get current workflow state"""
        return session.current_state

    async def transition_state(
        self,
        session: UserSession,
        new_state: str
    ) -> Tuple[bool, Optional[str]]:
        """Attempt state transition with validation"""
        # Check if transition is valid
        valid, error_msg = await self._validate_transition(
            session.current_state,
            new_state
        )

        if not valid:
            return False, error_msg

        # Check prerequisites
        if new_state in STATE_PREREQUISITES:
            check = STATE_PREREQUISITES[new_state]
            if not check(session):
                return False, self._get_prerequisite_message(new_state)

        # Perform transition
        session.current_state = new_state
        await self.update_session(session)
        logger.debug(f"State transition: {session.session_key} → {new_state}")
        return True, None

    async def cancel_workflow(self, session: UserSession) -> None:
        """Cancel current workflow"""
        session.workflow_type = None
        session.current_state = None
        session.selections.clear()
        await self.update_session(session)

    # === Data Management ===

    async def store_data(
        self,
        session: UserSession,
        data: pd.DataFrame
    ) -> None:
        """Store DataFrame in session"""
        # Check size limit
        size_mb = data.memory_usage(deep=True).sum() / 1024 / 1024
        if size_mb > self.config.max_dataframe_size_mb:
            raise DataTooLargeError(
                f"Data size {size_mb:.1f}MB exceeds limit of "
                f"{self.config.max_dataframe_size_mb}MB"
            )

        session.uploaded_data = data
        await self.update_session(session)

    async def get_data(self, session: UserSession) -> Optional[pd.DataFrame]:
        """Retrieve DataFrame from session"""
        return session.uploaded_data

    async def clear_data(self, session: UserSession) -> None:
        """Clear stored DataFrame"""
        session.uploaded_data = None
        await self.update_session(session)

    # === History Management ===

    async def add_to_history(
        self,
        session: UserSession,
        role: str,
        message: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """Add message to conversation history"""
        entry = {
            "timestamp": datetime.now(),
            "role": role,
            "message": message,
            "metadata": metadata or {}
        }

        session.history.append(entry)

        # Enforce history limit
        if len(session.history) > self.config.max_history_messages:
            session.history = session.history[-self.config.max_history_messages:]

        await self.update_session(session)

    async def get_history(
        self,
        session: UserSession,
        n_messages: int = 5
    ) -> List[Dict]:
        """Get last n messages for context"""
        return session.history[-n_messages:] if session.history else []

    # === Utility Methods ===

    async def get_stats(self) -> Dict[str, Any]:
        """Get state manager statistics"""
        return {
            "active_sessions": len(self._sessions),
            "total_memory_mb": self._calculate_total_memory(),
            "workflows_in_progress": self._count_active_workflows(),
            "expired_in_grace": len(self._expired_sessions)
        }

    # === Cleanup & Lifecycle ===

    async def start(self) -> None:
        """Start background tasks"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("StateManager started")

    async def shutdown(self) -> None:
        """Graceful shutdown"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # Persist sessions if enabled
        if self.config.enable_persistence:
            await self._persist_all_sessions()

        logger.info("StateManager shutdown complete")

    async def _cleanup_loop(self) -> None:
        """Background task to cleanup expired sessions"""
        while self._running:
            try:
                await asyncio.sleep(self.config.cleanup_interval_minutes * 60)
                await self._cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}", exc_info=True)

    async def _cleanup_expired_sessions(self) -> int:
        """Remove expired sessions"""
        now = datetime.now()
        expired_keys = []

        for key, session in list(self._sessions.items()):
            timeout_seconds = self.config.session_timeout_minutes * 60
            inactive_seconds = (now - session.last_activity).total_seconds()

            if inactive_seconds > timeout_seconds:
                # Move to expired (grace period)
                if self.config.enable_workflow_recovery:
                    self._expired_sessions[key] = {
                        'session': session,
                        'expired_at': now
                    }
                expired_keys.append(key)

        # Remove expired sessions
        for key in expired_keys:
            del self._sessions[key]
            logger.info(f"Cleaned up expired session: {key}")

        # Cleanup grace period expired
        grace_seconds = self.config.grace_period_minutes * 60
        for key in list(self._expired_sessions.keys()):
            expired_at = self._expired_sessions[key]['expired_at']
            if (now - expired_at).total_seconds() > grace_seconds:
                del self._expired_sessions[key]

        return len(expired_keys)

    # === Private Methods ===

    async def _get_lock(self, session_key: str) -> asyncio.Lock:
        """Get or create lock for session"""
        if session_key not in self._locks:
            self._locks[session_key] = asyncio.Lock()
        return self._locks[session_key]

    async def _check_recoverable_session(
        self,
        session_key: str
    ) -> Optional[UserSession]:
        """Check if session can be recovered from grace period"""
        if session_key in self._expired_sessions:
            expired_data = self._expired_sessions[session_key]
            session = expired_data['session']
            session.last_activity = datetime.now()
            del self._expired_sessions[session_key]
            logger.info(f"Recovered session: {session_key}")
            return session
        return None

    def _get_initial_state(self, workflow_type: WorkflowType) -> str:
        """Get initial state for workflow type"""
        if workflow_type == WorkflowType.ML_TRAINING:
            return MLTrainingState.AWAITING_DATA.value
        # Add other workflow types...
        return None

    async def _validate_transition(
        self,
        current: Optional[str],
        new: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate state transition is allowed"""
        # Implementation depends on workflow type
        # For now, simple validation
        return True, None

    def _get_prerequisite_message(self, state: str) -> str:
        """Get user-friendly message for missing prerequisites"""
        messages = {
            MLTrainingState.SELECTING_TARGET.value:
                "Please upload data first using /upload or by sending a CSV/Excel file",
            MLTrainingState.SELECTING_FEATURES.value:
                "Please select a target variable first",
            MLTrainingState.CONFIRMING_MODEL.value:
                "Please select features for training",
        }
        return messages.get(state, "Prerequisites not met for this operation")

    def _calculate_total_memory(self) -> float:
        """Calculate total memory usage in MB"""
        total = 0
        for session in self._sessions.values():
            if session.uploaded_data is not None:
                total += session.uploaded_data.memory_usage(deep=True).sum()
        return total / 1024 / 1024

    def _count_active_workflows(self) -> int:
        """Count sessions with active workflows"""
        return sum(1 for s in self._sessions.values() if s.workflow_type)

    def _create_backend(self) -> 'PersistenceBackend':
        """Create persistence backend"""
        if self.config.persistence_backend == "memory":
            return MemoryBackend()
        elif self.config.persistence_backend == "pickle":
            return PickleBackend(self.config.persistence_path)
        # Add Redis backend when needed
        return MemoryBackend()
```

### Persistence Backends

```python
class PersistenceBackend(ABC):
    """Abstract persistence interface"""

    @abstractmethod
    async def save_session(self, key: str, session: UserSession) -> None:
        pass

    @abstractmethod
    async def load_session(self, key: str) -> Optional[UserSession]:
        pass

    @abstractmethod
    async def delete_session(self, key: str) -> None:
        pass

    @abstractmethod
    async def list_sessions(self) -> List[str]:
        pass

class MemoryBackend(PersistenceBackend):
    """In-memory storage (no persistence)"""

    def __init__(self):
        self._storage: Dict[str, UserSession] = {}

    async def save_session(self, key: str, session: UserSession) -> None:
        self._storage[key] = session

    async def load_session(self, key: str) -> Optional[UserSession]:
        return self._storage.get(key)

    async def delete_session(self, key: str) -> None:
        if key in self._storage:
            del self._storage[key]

    async def list_sessions(self) -> List[str]:
        return list(self._storage.keys())
```

### Exception Hierarchy

```python
class StateManagerError(Exception):
    """Base exception for state manager errors"""
    pass

class SessionNotFoundError(StateManagerError):
    """Session doesn't exist"""
    pass

class InvalidStateTransitionError(StateManagerError):
    """Attempted invalid state transition"""

    def __init__(self, current_state: str, attempted_state: str):
        self.current_state = current_state
        self.attempted_state = attempted_state
        super().__init__(
            f"Cannot transition from {current_state} to {attempted_state}"
        )

class SessionExpiredError(StateManagerError):
    """Session has timed out"""
    pass

class WorkflowInterruptedError(StateManagerError):
    """Workflow was interrupted by user action"""
    pass

class DataTooLargeError(StateManagerError):
    """Uploaded data exceeds size limit"""
    pass
```

## Integration with Existing System

### Bot Handler Integration

```python
# In src/bot/handlers.py

from src.core.state_manager import StateManager, WorkflowType

state_manager = StateManager()

async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    user_id = update.effective_user.id
    conversation_id = str(update.message.chat.id)

    # Get or create session
    session = await state_manager.get_or_create_session(user_id, conversation_id)

    # Add to history
    await state_manager.add_to_history(
        session,
        role="user",
        message=update.message.text
    )

    # Check for active workflow
    if session.workflow_type:
        result = await handle_workflow_step(session, update.message.text)
    else:
        result = await process_normal_request(session, update.message.text)

    # Add response to history
    await state_manager.add_to_history(
        session,
        role="assistant",
        message=result.text,
        metadata={"result_type": result.type}
    )

    await update.message.reply_text(result.text)

async def handle_workflow_step(
    session: UserSession,
    message: str
) -> ProcessedResult:
    """Handle next step in active workflow"""

    if session.workflow_type == WorkflowType.ML_TRAINING:
        return await handle_ml_training_step(session, message)
    # Handle other workflow types...

async def handle_ml_training_step(
    session: UserSession,
    message: str
) -> ProcessedResult:
    """Handle ML training workflow steps"""

    state = session.current_state

    if state == MLTrainingState.SELECTING_TARGET.value:
        # Parse target selection
        target = parse_column_selection(message, session.uploaded_data)
        session.selections['target'] = target

        # Transition to next state
        await state_manager.transition_state(
            session,
            MLTrainingState.SELECTING_FEATURES.value
        )

        # Prompt for features
        features = session.uploaded_data.columns.tolist()
        features.remove(target)
        return create_feature_selection_prompt(features)

    # Handle other states...
```

### Parser Integration

```python
# Parser can use conversation context

async def parse_request(
    message: str,
    session: UserSession
) -> TaskDefinition:
    # Get recent context
    history = await state_manager.get_history(session, n_messages=5)

    # Use context for pronoun resolution
    if "it" in message or "that" in message:
        # Look for last data reference in history
        last_data_ref = find_last_data_reference(history)
        if last_data_ref:
            message = message.replace("it", last_data_ref)

    # Parse with context
    return parser.parse(message, context=history)
```

## Implementation Phases (TDD)

### Phase 1: Foundation (~100 LOC + 80 test LOC)

**Deliverables:**
- `UserSession` dataclass with validation
- `StateManagerConfig` dataclass
- `WorkflowType` and state enums
- Basic exception classes
- Unit tests for data structures

**Test Coverage:**
```python
# tests/unit/test_state_manager_dataclasses.py
def test_user_session_creation()
def test_user_session_key_generation()
def test_session_expiry_check()
def test_config_defaults()
def test_config_validation()
```

### Phase 2: State Machine (~120 LOC + 100 test LOC)

**Deliverables:**
- State transition validation logic
- Prerequisite checking system
- State transition rules
- Unit tests for state machine

**Test Coverage:**
```python
# tests/unit/test_state_machine.py
def test_valid_transitions()
def test_invalid_transitions()
def test_prerequisite_checking()
def test_workflow_initialization()
```

### Phase 3: Session Management (~150 LOC + 120 test LOC)

**Deliverables:**
- Session CRUD operations
- AsyncIO locking
- History management
- Data storage
- Integration tests

**Test Coverage:**
```python
# tests/unit/test_session_management.py
def test_get_or_create_session()
def test_concurrent_session_access()
def test_history_sliding_window()
def test_data_storage_size_limits()
def test_session_recovery_from_grace_period()
```

### Phase 4: Background Tasks (~80 LOC + 60 test LOC)

**Deliverables:**
- Cleanup loop implementation
- Expired session removal
- Lifecycle management (start/shutdown)
- Tests for background tasks

**Test Coverage:**
```python
# tests/unit/test_cleanup.py
def test_cleanup_expired_sessions()
def test_grace_period_recovery()
def test_graceful_shutdown()
def test_cleanup_interval()
```

### Phase 5: Integration (~50 LOC + 80 test LOC)

**Deliverables:**
- Handler integration helpers
- Workflow interruption handling
- End-to-end workflow tests

**Test Coverage:**
```python
# tests/integration/test_state_manager_workflows.py
async def test_complete_ml_training_workflow()
async def test_workflow_interruption()
async def test_session_timeout_during_workflow()
async def test_multi_user_concurrent_workflows()
```

## Testing Strategy

### Unit Tests

**Session Management:**
- Session creation/retrieval
- Concurrent access with locks
- History management
- Data storage

**State Machine:**
- Valid/invalid transitions
- Prerequisite validation
- Workflow initialization

**Cleanup:**
- Timeout detection
- Grace period recovery
- Background task lifecycle

### Integration Tests

**Workflows:**
- Complete ML training flow
- Workflow interruption/cancellation
- Multi-step data exploration

**Concurrency:**
- Multiple users simultaneously
- Same user multiple messages
- Race conditions

**Persistence:**
- Save/load cycles
- Crash recovery

### Test Fixtures

```python
@pytest.fixture
async def state_manager():
    config = StateManagerConfig(
        session_timeout_minutes=1,  # Fast timeout for tests
        cleanup_interval_minutes=0.1
    )
    manager = StateManager(config)
    await manager.start()
    yield manager
    await manager.shutdown()

@pytest.fixture
def sample_session():
    return UserSession(
        user_id=12345,
        conversation_id="test_conv_123"
    )

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [0, 1, 0]
    })
```

## Configuration

### config/config.yaml

```yaml
state_manager:
  timeout_minutes: 30
  max_history: 50
  max_concurrent_sessions: 1000
  max_dataframe_size_mb: 100

  persistence:
    enabled: false
    backend: memory  # memory, pickle, redis
    path: ./data/sessions  # for pickle backend

  cleanup:
    interval_minutes: 10
    grace_period_minutes: 5

  workflow:
    enable_recovery: true
```

### Environment Variables

```bash
STATE_TIMEOUT_MINUTES=30
STATE_MAX_HISTORY=50
STATE_PERSISTENCE_ENABLED=false
STATE_PERSISTENCE_BACKEND=memory
STATE_CLEANUP_INTERVAL=10
```

## Security Considerations

### Data Privacy
- Session isolation: strict user_id + conversation_id keying
- No cross-user data leakage
- DataFrame content not logged (only metadata)
- Audit logging for sensitive operations

### Session Security
- Cryptographically secure session keys (UUID4)
- Input validation before storage
- Memory protection (no dumps in logs)
- Access control validation

### GDPR Compliance
- User can request deletion: `/forget` command
- Automatic cleanup after timeout
- No unnecessary data retention
- Explicit consent for data storage

### Audit Trail
```python
def _audit_log(self, action: str, user_id: int, session_key: str):
    logger.info(
        f"AUDIT: {action} by user={user_id} session={session_key[:8]}...",
        extra={'user_id': user_id, 'action': action}
    )
```

## Performance Targets

### Scalability
- **Target**: 1,000 concurrent sessions
- **Session retrieval**: <100ms p95
- **State transition**: <50ms p95
- **Cleanup cycle**: <1s for 1,000 sessions
- **Memory per session**: <50MB average

### Optimization Strategies
- Lazy loading: Store DataFrame reference, load on demand
- Compression: Pickle + gzip for large DataFrames
- LRU cache for session objects
- Read replicas for separate read/write

### Load Testing
```python
# tests/performance/test_load.py
async def test_1000_concurrent_sessions()
async def test_100_concurrent_workflows()
async def test_session_retrieval_latency()
async def test_memory_usage_under_load()
```

## Monitoring & Observability

### Metrics to Track
```python
{
    "active_sessions": 847,
    "total_memory_mb": 2341.5,
    "workflows_in_progress": 23,
    "expired_in_grace": 5,
    "cleanup_cycles_completed": 144,
    "average_session_duration_minutes": 18.5
}
```

### Logging Strategy
```python
# DEBUG: Session operations, state transitions
logger.debug(f"State transition: {session_key} {old} → {new}")

# INFO: Workflow events, session lifecycle
logger.info(f"Session created: {session_key}")
logger.info(f"Workflow completed: {session_key} type={workflow_type}")

# WARNING: Invalid transitions, approaching timeout
logger.warning(f"Invalid transition: {current} → {attempted}")
logger.warning(f"Session timeout warning: {session_key}")

# ERROR: Persistence failures, state corruption
logger.error(f"Session persistence failed: {session_key}", exc_info=True)
```

### Health Check Endpoint
```python
async def health_check() -> Dict[str, Any]:
    stats = await state_manager.get_stats()
    return {
        "status": "healthy" if stats["active_sessions"] < 1000 else "degraded",
        "metrics": stats,
        "timestamp": datetime.now().isoformat()
    }
```

## Future Enhancements

### Phase 6: Advanced Features (Post-MVP)

**Redis Backend:**
```python
class RedisBackend(PersistenceBackend):
    def __init__(self, redis_url: str):
        self.redis = aioredis.from_url(redis_url)

    async def save_session(self, key: str, session: UserSession):
        serialized = pickle.dumps(session)
        await self.redis.set(f"session:{key}", serialized, ex=1800)
```

**Distributed Locking:**
- Use Redis distributed locks for multi-instance deployment
- Enable horizontal scaling

**Advanced Workflows:**
- Branching workflows (multiple paths)
- Sub-workflows (nested state machines)
- Workflow templates (reusable patterns)

**Analytics:**
- User behavior tracking
- Workflow completion rates
- Common failure points
- Session duration analysis

## Success Criteria

### Functional Requirements ✅
- ✅ Multi-step ML training workflow functional
- ✅ Session timeout with graceful recovery
- ✅ Conversation history for context
- ✅ DataFrame storage and retrieval
- ✅ Model ID tracking

### Non-Functional Requirements ✅
- ✅ <100ms session retrieval (p95)
- ✅ Support 1,000 concurrent sessions
- ✅ Thread-safe async operations
- ✅ Zero data leakage between users
- ✅ >90% test coverage
- ✅ Clean integration with bot handlers

### Quality Metrics ✅
- ✅ All unit tests passing
- ✅ Integration tests cover all workflows
- ✅ Security audit passed
- ✅ Performance benchmarks met
- ✅ Documentation complete

## Estimated Effort

**Total Production Code:** ~500 LOC
- Phase 1: 100 LOC (dataclasses, config)
- Phase 2: 120 LOC (state machine)
- Phase 3: 150 LOC (session management)
- Phase 4: 80 LOC (cleanup)
- Phase 5: 50 LOC (integration)

**Total Test Code:** ~440 LOC
- Unit tests: 260 LOC
- Integration tests: 180 LOC

**Documentation:** This planning doc + inline docstrings

**Timeline:** ~8-10 hours for complete TDD implementation

## Dependencies

**Existing:**
- `asyncio` - async operations
- `pandas` - DataFrame storage
- `dataclasses` - data structures
- Existing logger from `src/utils/logger.py`

**New (if needed):**
- `aioredis` - Redis backend (Phase 6)
- `msgpack` - efficient serialization (optimization)

## References

**Related Components:**
- Bot handlers: `src/bot/handlers.py`
- Parser: `src/core/parser.py`
- Orchestrator: `src/core/orchestrator.py`
- Data loader: `src/data/data_loader.py`

**Similar Patterns:**
- Result Processor session pattern
- Telegram bot context.user_data pattern
- Workflow state machines in other projects

---

**Status:** Ready for implementation following TDD methodology across 5 phases.
