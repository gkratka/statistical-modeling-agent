# Integration Layer - Implementation Plan

**Status:** 📋 **PLANNING**
**Date:** 2025-10-01
**Branch:** `feature/integration-layer`
**Dependencies:** State Manager (✅ merged), ML Engine, Stats Engine, Data Loader, Result Processor, Orchestrator

---

## Executive Summary

**Goal:** Build a comprehensive integration layer that connects the State Manager, Parser, Orchestrator, ML/Stats Engines, and Result Processor to the Telegram bot for multi-step conversation workflows.

**Key Challenge:** The existing orchestrator has its own StateManager that conflicts with the newly merged comprehensive StateManager from `src/core/state_manager.py`.

**Approach:** Refactor orchestrator to use the new StateManager, create integration components, and add command handlers for `/train`, `/predict`, `/stats`, `/cancel`.

**Estimated Effort:** ~3,300 LOC (950 production + 1,850 tests + 500 modifications) over 3-4 weeks

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Current State Analysis](#current-state-analysis)
3. [Phase 1: Foundation Components](#phase-1-foundation-components)
4. [Phase 2: Orchestrator Refactoring](#phase-2-orchestrator-refactoring)
5. [Phase 3: Handler Updates](#phase-3-handler-updates)
6. [Phase 4: Command Handlers](#phase-4-command-handlers)
7. [Phase 5: Configuration & Documentation](#phase-5-configuration--documentation)
8. [Testing Strategy](#testing-strategy)
9. [Implementation Schedule](#implementation-schedule)
10. [Risk Mitigation](#risk-mitigation)
11. [Success Criteria](#success-criteria)

---

## Architecture Overview

### System Flow Diagram

```
┌─────────────┐
│ Telegram    │
│ User        │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────┐
│ handlers.py                                         │
│ ┌─────────────┐ ┌──────────────┐ ┌──────────────┐ │
│ │ /start      │ │ /train       │ │ message_     │ │
│ │ /help       │ │ /predict     │ │ handler      │ │
│ │ /diagnostic │ │ /stats       │ │ document_    │ │
│ │             │ │ /cancel      │ │ handler      │ │
│ └─────────────┘ └──────────────┘ └──────────────┘ │
└───────────┬─────────────────────────────────────────┘
            │
            ▼
┌────────────────────────────────────────────────────┐
│ IntegrationLayer                                   │
│ - Session management                               │
│ - Workflow state checking                          │
│ - Message routing                                  │
│ - Data coordination                                │
└───┬────────────────────────────────────────────────┘
    │
    ├─► StateManager (src/core/state_manager.py)
    │   - Session CRUD
    │   - Workflow start/transition/cancel
    │   - Data storage
    │   - History tracking
    │
    ├─► WorkflowHandler
    │   - Multi-step flow management
    │   - User input parsing (columns, model types)
    │   - State-specific prompts
    │   - Validation
    │
    └─► Parser → Orchestrator → Engine → ResultProcessor
        - Task definition creation
        - Task execution
        - Result formatting
        - Response building
```

### Data Flow: ML Training Workflow

```
User: /train
  ↓
Handler → IntegrationLayer.get_or_create_session()
  ↓
StateManager.start_workflow(WorkflowType.ML_TRAINING)
  ↓
Session.current_state = AWAITING_DATA
  ↓
Response: "Please upload your training data"

User: [uploads CSV]
  ↓
document_handler → DataLoader.load_from_telegram()
  ↓
StateManager.store_data(session, dataframe)
  ↓
StateManager.transition_state(session, SELECTING_TARGET)
  ↓
Response: "Select target variable from: [age, income, education, ...]"

User: "predict income"
  ↓
WorkflowHandler.parse_target_selection()
  ↓
session.selections['target'] = 'income'
  ↓
StateManager.transition_state(session, SELECTING_FEATURES)
  ↓
Response: "Select features from: [age, education, hours_per_week, ...]"

User: "use age, education, hours_per_week"
  ↓
WorkflowHandler.parse_feature_selection()
  ↓
session.selections['features'] = ['age', 'education', 'hours_per_week']
  ↓
StateManager.transition_state(session, CONFIRMING_MODEL)
  ↓
Response: "Choose model: neural_network, random_forest, linear_regression"

User: "random forest"
  ↓
WorkflowHandler.parse_model_type()
  ↓
session.selections['model_type'] = 'random_forest'
  ↓
StateManager.transition_state(session, TRAINING)
  ↓
Parser.create_ml_task() → Orchestrator.execute_task() → MLEngine.train()
  ↓
StateManager.transition_state(session, COMPLETE)
  ↓
Response: "✅ Model trained! Accuracy: 85.3%, F1: 0.82"
```

---

## Current State Analysis

### Existing Components

**✅ Working:**
1. **State Manager** (`src/core/state_manager.py`) - Just merged, 71 passing tests
2. **ML Engine** (`src/engines/ml_engine.py`) - Neural networks, RF, linear regression
3. **Stats Engine** (`src/engines/stats_engine.py`) - Descriptive stats, correlations
4. **Data Loader** (`src/processors/data_loader.py`) - CSV/Excel loading
5. **Result Processor** (`src/processors/result_processor.py`) - Result formatting
6. **Parser** (`src/core/parser.py`) - Natural language parsing
7. **Basic Handlers** (`src/bot/handlers.py`) - /start, /help, document upload

**⚠️ Conflicts:**
1. **Orchestrator** (`src/core/orchestrator.py`) has its own StateManager (lines 64-127)
2. **WorkflowState enum** in orchestrator conflicts with new state machine
3. **ConversationState** dataclass duplicates UserSession functionality

### Integration Gaps

**Missing:**
1. ❌ No State Manager integration in handlers
2. ❌ No /train, /predict, /stats, /cancel commands
3. ❌ No multi-step workflow handling
4. ❌ No session-based data storage
5. ❌ No workflow state checking in message routing
6. ❌ No integration layer to coordinate components

---

## Phase 1: Foundation Components

**Duration:** 1 week
**Dependencies:** None (State Manager already merged)

### 1.1 IntegrationLayer Class

**File:** `src/bot/integration_layer.py` (~300 LOC)

**Purpose:** Central coordination layer that bridges handlers, State Manager, and other components.

**Responsibilities:**
- Initialize and manage StateManager instance
- Session retrieval/creation for each message
- Workflow state detection and routing
- Coordinate data storage between context.user_data and StateManager
- Handle exceptions and format user-friendly errors

**Class Structure:**
```python
class IntegrationLayer:
    """
    Integration layer coordinating Telegram handlers with State Manager,
    Parser, Orchestrator, and other core components.
    """

    def __init__(
        self,
        state_manager: StateManager,
        orchestrator: Optional[TaskOrchestrator] = None,
        parser: Optional[RequestParser] = None
    ):
        """Initialize integration layer with dependencies."""
        self.state_manager = state_manager
        self.orchestrator = orchestrator or TaskOrchestrator()
        self.parser = parser or RequestParser()
        self.logger = get_logger(__name__)

    async def get_or_create_session(
        self,
        user_id: int,
        conversation_id: str
    ) -> UserSession:
        """
        Get or create user session.

        This is the primary entry point for all message handling.
        Ensures every user interaction has an associated session.
        """
        return await self.state_manager.get_or_create_session(
            user_id, conversation_id
        )

    async def check_workflow_state(
        self,
        session: UserSession
    ) -> Optional[WorkflowType]:
        """
        Check if user is in active workflow.

        Returns:
            WorkflowType if active, None if idle
        """
        return session.workflow_type

    async def route_message(
        self,
        session: UserSession,
        message_text: str
    ) -> HandlerType:
        """
        Determine which handler should process this message.

        Returns:
            - WORKFLOW_HANDLER if in active workflow
            - STATS_HANDLER for stats queries
            - ML_HANDLER for ML-related requests
            - GENERAL_HANDLER for other messages
        """
        if session.workflow_type:
            return HandlerType.WORKFLOW

        # Simple keyword detection
        if any(word in message_text.lower() for word in ['mean', 'std', 'correlation', 'stats']):
            return HandlerType.STATS

        if any(word in message_text.lower() for word in ['train', 'model', 'predict']):
            return HandlerType.ML

        return HandlerType.GENERAL

    async def store_uploaded_data(
        self,
        session: UserSession,
        dataframe: pd.DataFrame,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Store uploaded data in session.

        Integrates DataLoader results with State Manager storage.
        """
        await self.state_manager.store_data(session, dataframe)

        # Add metadata to session history
        await self.state_manager.add_to_history(
            session,
            role="system",
            message=f"Data uploaded: {metadata.get('shape', (0, 0))}"
        )

    async def handle_workflow_message(
        self,
        session: UserSession,
        message_text: str
    ) -> str:
        """
        Handle message when user is in active workflow.

        Routes to WorkflowHandler for state-specific processing.
        """
        from src.bot.workflow_handler import WorkflowHandler

        workflow_handler = WorkflowHandler(
            self.state_manager,
            self.orchestrator,
            self.parser
        )

        return await workflow_handler.process_message(session, message_text)

    async def format_error(
        self,
        exception: Exception,
        session: Optional[UserSession] = None
    ) -> str:
        """
        Format exception into user-friendly error message.

        Provides contextual help based on error type and current state.
        """
        from src.bot.response_builder import ResponseBuilder

        builder = ResponseBuilder()
        return builder.format_state_error(exception, session)

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions. Returns count of cleaned sessions."""
        return await self.state_manager.cleanup_expired_sessions()
```

**Error Handling:**
```python
class IntegrationLayerError(AgentError):
    """Base exception for integration layer errors."""
    pass

class SessionRetrievalError(IntegrationLayerError):
    """Failed to retrieve or create session."""
    pass

class WorkflowRoutingError(IntegrationLayerError):
    """Failed to route message to appropriate handler."""
    pass
```

### 1.2 ResponseBuilder Class

**File:** `src/bot/response_builder.py` (~250 LOC)

**Purpose:** Generate user-friendly messages, prompts, and error responses.

**Responsibilities:**
- Build workflow state prompts
- Format error messages from State Manager exceptions
- Generate column selection interfaces
- Create progress indicators
- Format prerequisite violation messages

**Class Structure:**
```python
class ResponseBuilder:
    """
    Builds user-friendly response messages for Telegram bot.

    Handles formatting for:
    - Workflow prompts
    - Error messages
    - Progress indicators
    - Column selections
    - Validation errors
    """

    def build_workflow_prompt(
        self,
        workflow_state: str,
        session: UserSession
    ) -> str:
        """
        Generate contextual prompt for current workflow state.

        Examples:
        - SELECTING_TARGET: "Select target variable from: [col1, col2, ...]"
        - SELECTING_FEATURES: "Select feature columns (comma-separated): [col1, col2, ...]"
        - CONFIRMING_MODEL: "Choose model type: neural_network, random_forest, linear_regression"
        """
        from src.core.state_manager import MLTrainingState, MLPredictionState

        if workflow_state == MLTrainingState.SELECTING_TARGET.value:
            return self._build_target_prompt(session)
        elif workflow_state == MLTrainingState.SELECTING_FEATURES.value:
            return self._build_feature_prompt(session)
        elif workflow_state == MLTrainingState.CONFIRMING_MODEL.value:
            return self._build_model_prompt()

        return "Please continue with the workflow..."

    def format_state_error(
        self,
        error: Exception,
        session: Optional[UserSession] = None
    ) -> str:
        """
        Format State Manager exception into user-friendly message.

        Provides recovery instructions based on error type.
        """
        from src.utils.exceptions import (
            InvalidStateTransitionError,
            PrerequisiteNotMetError,
            SessionExpiredError,
            DataSizeLimitError
        )

        if isinstance(error, InvalidStateTransitionError):
            return self._format_invalid_transition(error, session)
        elif isinstance(error, PrerequisiteNotMetError):
            return self._format_prerequisite_error(error, session)
        elif isinstance(error, SessionExpiredError):
            return self._format_session_expired(error)
        elif isinstance(error, DataSizeLimitError):
            return self._format_data_size_error(error)

        return f"⚠️ **Error:** {str(error)}\n\nPlease try again or use /cancel."

    def format_column_selection(
        self,
        columns: List[str],
        prompt_type: Literal["target", "features"],
        max_display: int = 15
    ) -> str:
        """
        Format column list for user selection.

        Groups numeric and categorical columns, shows first N.
        """
        if len(columns) <= max_display:
            column_list = "\n".join(f"• {col}" for col in columns)
        else:
            shown = columns[:max_display]
            remaining = len(columns) - max_display
            column_list = "\n".join(f"• {col}" for col in shown)
            column_list += f"\n... and {remaining} more"

        if prompt_type == "target":
            return (
                f"🎯 **Select Target Variable**\n\n"
                f"Choose the column you want to predict:\n\n"
                f"{column_list}\n\n"
                f"**Example:** \"predict income\" or just \"income\""
            )
        else:
            return (
                f"📊 **Select Feature Columns**\n\n"
                f"Choose columns to use as features:\n\n"
                f"{column_list}\n\n"
                f"**Example:** \"use age, education, hours_per_week\""
            )

    def build_progress_indicator(
        self,
        current_step: str,
        workflow_type: WorkflowType
    ) -> str:
        """
        Build progress indicator showing workflow completion.

        Example: "Progress: ●●●○○ (3/5) - Selecting Features"
        """
        from src.core.state_manager import MLTrainingState

        if workflow_type == WorkflowType.ML_TRAINING:
            states = [
                MLTrainingState.AWAITING_DATA,
                MLTrainingState.SELECTING_TARGET,
                MLTrainingState.SELECTING_FEATURES,
                MLTrainingState.CONFIRMING_MODEL,
                MLTrainingState.TRAINING,
                MLTrainingState.COMPLETE
            ]

            try:
                current_idx = [s.value for s in states].index(current_step)
                filled = "●" * (current_idx + 1)
                empty = "○" * (len(states) - current_idx - 1)
                return f"Progress: {filled}{empty} ({current_idx + 1}/{len(states)})"
            except ValueError:
                return f"Progress: {current_step}"

        return ""

    def format_prerequisite_error(
        self,
        missing_prerequisites: List[str],
        session: UserSession
    ) -> str:
        """
        Format prerequisite violation error with recovery steps.

        Provides specific instructions based on what's missing.
        """
        prereq_names = {
            "uploaded_data": ("📁 Upload Data", "Please upload a CSV file with your data"),
            "target_selection": ("🎯 Select Target", "Choose which column to predict"),
            "feature_selection": ("📊 Select Features", "Choose which columns to use as features"),
            "model_type_selection": ("🤖 Choose Model", "Select model type: neural_network, random_forest, linear_regression"),
            "trained_model": ("🧠 Train Model", "You need a trained model first. Use /train"),
            "prediction_data": ("📁 Upload Data", "Upload data for predictions")
        }

        missing_items = [
            f"• {prereq_names.get(p, (p, p))[0]}: {prereq_names.get(p, (p, p))[1]}"
            for p in missing_prerequisites
        ]

        return (
            f"⚠️ **Cannot Proceed**\n\n"
            f"**Missing Prerequisites:**\n"
            + "\n".join(missing_items) +
            f"\n\n**Current State:** {session.current_state}\n"
            f"Please complete the missing steps first."
        )

    def _build_target_prompt(self, session: UserSession) -> str:
        """Build target selection prompt with available columns."""
        df = session.uploaded_data
        if df is None:
            return "⚠️ No data available. Please upload data first."

        columns = df.columns.tolist()
        return self.format_column_selection(columns, "target")

    def _build_feature_prompt(self, session: UserSession) -> str:
        """Build feature selection prompt, excluding target."""
        df = session.uploaded_data
        target = session.selections.get('target')

        if df is None:
            return "⚠️ No data available."

        # Exclude target from feature selection
        columns = [col for col in df.columns if col != target]
        return self.format_column_selection(columns, "features")

    def _build_model_prompt(self) -> str:
        """Build model selection prompt."""
        return (
            f"🤖 **Choose Model Type**\n\n"
            f"Available models:\n\n"
            f"• **neural_network** - Deep learning, best for complex patterns\n"
            f"• **random_forest** - Ensemble method, good for most tasks\n"
            f"• **linear_regression** - Simple and interpretable\n\n"
            f"**Example:** \"random forest\" or \"use neural network\""
        )
```

### 1.3 Tests

#### test_integration_layer.py (~300 LOC)
```python
import pytest
import pandas as pd
from src.bot.integration_layer import IntegrationLayer
from src.core.state_manager import StateManager, WorkflowType

@pytest.mark.asyncio
class TestIntegrationLayer:
    """Test IntegrationLayer functionality."""

    @pytest.fixture
    def integration_layer(self):
        """Create integration layer instance."""
        state_manager = StateManager()
        return IntegrationLayer(state_manager)

    async def test_get_or_create_session_new(self, integration_layer):
        """Test creating new session."""
        session = await integration_layer.get_or_create_session(123, "conv_1")

        assert session is not None
        assert session.user_id == 123
        assert session.conversation_id == "conv_1"

    async def test_check_workflow_state_idle(self, integration_layer):
        """Test workflow state check when idle."""
        session = await integration_layer.get_or_create_session(123, "conv_1")

        workflow = await integration_layer.check_workflow_state(session)

        assert workflow is None

    async def test_check_workflow_state_active(self, integration_layer):
        """Test workflow state check when in workflow."""
        session = await integration_layer.get_or_create_session(123, "conv_1")
        await integration_layer.state_manager.start_workflow(
            session, WorkflowType.ML_TRAINING
        )

        workflow = await integration_layer.check_workflow_state(session)

        assert workflow == WorkflowType.ML_TRAINING

    async def test_route_message_workflow(self, integration_layer):
        """Test message routing when in workflow."""
        session = await integration_layer.get_or_create_session(123, "conv_1")
        await integration_layer.state_manager.start_workflow(
            session, WorkflowType.ML_TRAINING
        )

        handler_type = await integration_layer.route_message(session, "income")

        assert handler_type == HandlerType.WORKFLOW

    async def test_route_message_stats(self, integration_layer):
        """Test message routing for stats query."""
        session = await integration_layer.get_or_create_session(123, "conv_1")

        handler_type = await integration_layer.route_message(
            session, "calculate mean and std"
        )

        assert handler_type == HandlerType.STATS

    async def test_store_uploaded_data(self, integration_layer):
        """Test data storage integration."""
        session = await integration_layer.get_or_create_session(123, "conv_1")
        df = pd.DataFrame({"x": [1, 2, 3]})
        metadata = {"shape": (3, 1)}

        await integration_layer.store_uploaded_data(session, df, metadata)

        assert session.uploaded_data is not None
        assert len(session.uploaded_data) == 3
```

#### test_response_builder.py (~200 LOC)
```python
import pytest
from src.bot.response_builder import ResponseBuilder
from src.core.state_manager import UserSession, WorkflowType, MLTrainingState

class TestResponseBuilder:
    """Test ResponseBuilder functionality."""

    @pytest.fixture
    def builder(self):
        """Create response builder instance."""
        return ResponseBuilder()

    def test_build_target_prompt(self, builder):
        """Test target selection prompt generation."""
        session = UserSession(
            user_id=123,
            conversation_id="conv_1",
            uploaded_data=pd.DataFrame({"a": [1], "b": [2], "c": [3]})
        )

        prompt = builder.build_workflow_prompt(
            MLTrainingState.SELECTING_TARGET.value, session
        )

        assert "Select Target Variable" in prompt
        assert "• a" in prompt
        assert "• b" in prompt
        assert "• c" in prompt

    def test_format_column_selection_with_many_columns(self, builder):
        """Test column selection with truncation."""
        columns = [f"col_{i}" for i in range(20)]

        result = builder.format_column_selection(columns, "target", max_display=10)

        assert "col_0" in result
        assert "col_9" in result
        assert "... and 10 more" in result

    def test_build_progress_indicator(self, builder):
        """Test progress indicator generation."""
        progress = builder.build_progress_indicator(
            MLTrainingState.SELECTING_FEATURES.value,
            WorkflowType.ML_TRAINING
        )

        assert "●●●" in progress  # 3 filled circles
        assert "○○○" in progress  # 3 empty circles
        assert "(3/6)" in progress
```

---

## Phase 2: Orchestrator Refactoring

**Duration:** 1-2 weeks
**Dependencies:** Phase 1

### 2.1 Remove Conflicting Components

**File:** `src/core/orchestrator.py`

**Remove these classes/enums:**

```python
# DELETE: Lines 31-43
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

# DELETE: Lines 46-62
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

# DELETE: Lines 64-127
class StateManager:
    """Manages conversation states with TTL-based cleanup."""
    # ... entire class ...
```

### 2.2 Update Imports

```python
# OLD IMPORTS (remove)
# from src.core.orchestrator import StateManager, ConversationState, WorkflowState

# NEW IMPORTS (add)
from src.core.state_manager import (
    StateManager,
    StateManagerConfig,
    UserSession,
    WorkflowType,
    MLTrainingState,
    MLPredictionState
)
```

### 2.3 Update TaskOrchestrator

**Current initialization (line 638):**
```python
def __init__(
    self,
    enable_logging: bool = True,
    data_loader: Optional[DataLoader] = None,
    state_ttl_minutes: int = 60,
    ml_config: Optional[MLEngineConfig] = None
) -> None:
    self.state_manager = StateManager(ttl_minutes=state_ttl_minutes)  # OLD
```

**New initialization:**
```python
def __init__(
    self,
    enable_logging: bool = True,
    data_loader: Optional[DataLoader] = None,
    state_config: Optional[StateManagerConfig] = None,
    ml_config: Optional[MLEngineConfig] = None
) -> None:
    # Use new StateManager
    self.state_manager = StateManager(
        config=state_config or StateManagerConfig()
    )
```

**Update execute_task method (lines 664-744):**

```python
# OLD
state = await self.state_manager.get_state(task.user_id, task.conversation_id)

# NEW
session = await self.state_manager.get_or_create_session(
    task.user_id, task.conversation_id
)

# OLD
if state.workflow_state != WorkflowState.IDLE:

# NEW
if session.workflow_type is not None:

# OLD
state.data_sources

# NEW
session.uploaded_data  # Direct DataFrame access
```

### 2.4 Update WorkflowEngine

**Current:** Uses old StateManager and WorkflowState
**New:** Use new StateManager and ML state enums

```python
class WorkflowEngine:
    """Manages multi-step workflows using new State Manager."""

    def __init__(self, state_manager: StateManager) -> None:
        self.state_manager = state_manager
        self.logger = logger

    async def start_ml_training(
        self,
        session: UserSession
    ) -> str:
        """Start ML training workflow."""
        await self.state_manager.start_workflow(
            session, WorkflowType.ML_TRAINING
        )

        # Session is now in AWAITING_DATA state
        return "ML training workflow started. Please upload your training data."

    async def advance_training_workflow(
        self,
        session: UserSession,
        new_state: str
    ) -> Tuple[bool, Optional[str], List[str]]:
        """Advance ML training workflow to next state."""
        return await self.state_manager.transition_state(session, new_state)

    async def cancel_workflow(
        self,
        session: UserSession
    ) -> None:
        """Cancel active workflow."""
        await self.state_manager.cancel_workflow(session)
```

### 2.5 Update DataManager

Keep DataManager but integrate with State Manager session storage:

```python
class DataManager:
    """Manages data lifecycle with State Manager integration."""

    async def load_and_store_data(
        self,
        telegram_file,
        file_name: str,
        file_size: int,
        session: UserSession,
        context
    ) -> None:
        """Load data and store in session."""
        # Use DataLoader
        dataframe, metadata = await self.data_loader.load_from_telegram(
            telegram_file, file_name, file_size, context
        )

        # Store in State Manager
        await self.state_manager.store_data(session, dataframe)

        self.logger.info(f"Loaded and stored data: {dataframe.shape}")
```

### 2.6 Tests

Update existing orchestrator tests:
- Replace `ConversationState` with `UserSession`
- Replace `WorkflowState` with `MLTrainingState`/`MLPredictionState`
- Update state manager method calls to new API

Add new integration tests:
```python
@pytest.mark.asyncio
class TestOrchestratorStateIntegration:
    """Test orchestrator integration with new State Manager."""

    async def test_orchestrator_uses_state_manager(self):
        """Test orchestrator properly uses new State Manager."""
        orchestrator = TaskOrchestrator()

        # Verify state manager is correct type
        assert isinstance(orchestrator.state_manager, StateManager)
        assert hasattr(orchestrator.state_manager, 'get_or_create_session')

    async def test_execute_task_with_session(self):
        """Test task execution uses session-based state."""
        orchestrator = TaskOrchestrator()
        task = TaskDefinition(
            user_id=123,
            conversation_id="conv_1",
            task_type="stats",
            operation="mean",
            parameters={"columns": ["x"]}
        )
        df = pd.DataFrame({"x": [1, 2, 3]})

        result = await orchestrator.execute_task(task, df)

        assert result["success"]
        # Verify session was created
        session = await orchestrator.state_manager.get_session(123, "conv_1")
        assert session is not None
```

---

## Phase 3: Handler Updates

**Duration:** 1 week
**Dependencies:** Phase 2

### 3.1 Initialize Global Components

**File:** `src/bot/handlers.py`

Add to top of file (after imports):

```python
# Integration layer components
from src.core.state_manager import StateManager, StateManagerConfig
from src.bot.integration_layer import IntegrationLayer
from src.bot.response_builder import ResponseBuilder

# Initialize global instances
state_manager = StateManager(StateManagerConfig(
    session_timeout_minutes=30,
    grace_period_minutes=5,
    max_data_size_mb=100,
    max_history_messages=50
))

integration_layer = IntegrationLayer(state_manager)
response_builder = ResponseBuilder()

logger.info("Integration layer initialized with State Manager")
```

### 3.2 Update message_handler

**Current (lines 117-256):** Basic orchestrator integration
**Update:** Add workflow state checking

```python
@telegram_handler
@log_user_action("Message received")
async def message_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle regular text messages with State Manager integration.

    Routes messages based on workflow state and content.
    """
    user_id = update.effective_user.id
    conversation_id = str(update.effective_chat.id)
    message_text = update.message.text or ""

    try:
        # Get or create session
        session = await integration_layer.get_or_create_session(
            user_id, conversation_id
        )

        # Add message to history
        await state_manager.add_to_history(
            session, role="user", message=message_text
        )

        # Check if user is in active workflow
        workflow_type = await integration_layer.check_workflow_state(session)

        if workflow_type:
            # Handle workflow-specific message
            logger.info(f"Processing workflow message: {workflow_type.value}")
            response = await integration_layer.handle_workflow_message(
                session, message_text
            )
            await update.message.reply_text(response, parse_mode="Markdown")
            return

        # Check if user has data for non-workflow processing
        user_data = safe_get_user_data(context, user_id)

        if not user_data and not session.uploaded_data:
            response_message = Messages.UPLOAD_DATA_FIRST
            await update.message.reply_text(response_message, parse_mode="Markdown")
            return

        # Get dataframe (prefer session storage, fallback to context)
        dataframe = session.uploaded_data or user_data.get('dataframe')
        metadata = user_data.get('metadata', {}) if user_data else {}

        # Handle data info requests
        if any(word in message_text.lower() for word in ['column', 'columns', 'what data']):
            columns = list(dataframe.columns) if dataframe is not None else []
            response_message = (
                f"📊 **Your Data**\n\n"
                f"**Columns ({len(columns)}):**\n"
                + "\n".join(f"• {col}" for col in columns[:10])
                + (f"\n... and {len(columns) - 10} more" if len(columns) > 10 else "")
                + f"\n\n**Shape:** {dataframe.shape[0]:,} rows × {dataframe.shape[1]} columns"
            )
            await update.message.reply_text(response_message, parse_mode="Markdown")
            return

        # Process through parser → orchestrator → formatter
        try:
            from src.core.parser import RequestParser
            from src.core.orchestrator import TaskOrchestrator
            from src.utils.result_formatter import TelegramResultFormatter

            parser = RequestParser()
            orchestrator = TaskOrchestrator()
            formatter = TelegramResultFormatter()

            # Parse request
            task = parser.parse_request(
                text=message_text,
                user_id=user_id,
                conversation_id=conversation_id,
                data_source=None
            )

            # Execute task
            result = await orchestrator.execute_task(task, dataframe)

            # Format result
            if task.task_type == "script":
                from src.bot.script_handler import ScriptHandler
                script_handler = ScriptHandler(parser, orchestrator)
                response_message = script_handler.format_script_results(result)
            else:
                response_message = formatter.format_stats_result(result)

            await update.message.reply_text(response_message, parse_mode="Markdown")

        except ParseError as e:
            response_message = (
                f"❓ **Request Not Understood**\n\n"
                f"I couldn't understand: \"{message_text}\"\n\n"
                f"**Try:**\n"
                f"• \"Calculate mean for age\"\n"
                f"• \"Show correlation matrix\"\n"
                f"• Use /train for ML model training"
            )
            await update.message.reply_text(response_message, parse_mode="Markdown")

        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            error_response = await integration_layer.format_error(e, session)
            await update.message.reply_text(error_response, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Critical error in message handler: {e}", exc_info=True)
        await update.message.reply_text(
            "⚠️ An unexpected error occurred. Please try again.",
            parse_mode="Markdown"
        )
```

### 3.3 Update document_handler

**Current (lines 355-468):** Uses DataLoader, stores in context.user_data
**Update:** Integrate with State Manager

```python
@telegram_handler
@log_user_action("File upload")
async def document_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Handle document uploads with State Manager integration.
    """
    if not update.message.document:
        logger.error("Received document message without document attachment")
        return

    user_id = update.effective_user.id
    conversation_id = str(update.effective_chat.id)
    document = update.message.document
    file_name = document.file_name or "unknown"
    file_size = document.file_size or 0

    try:
        # Get or create session
        session = await integration_layer.get_or_create_session(
            user_id, conversation_id
        )

        # Send processing message
        processing_msg = await update.message.reply_text(
            Messages.PROCESSING_FILE,
            parse_mode="Markdown"
        )

        # Get file from Telegram
        file_obj = await context.bot.get_file(document.file_id)

        # Load through DataLoader
        from src.processors.data_loader import DataLoader
        loader = DataLoader()
        df, metadata = await loader.load_from_telegram(
            file_obj, file_name, file_size, context
        )

        # Store in State Manager
        await integration_layer.store_uploaded_data(session, df, metadata)

        # Also store in context.user_data for backward compatibility
        if not hasattr(context, 'user_data'):
            context.user_data = {}
        context.user_data[f'data_{user_id}'] = {
            'dataframe': df,
            'metadata': metadata,
            'file_name': file_name
        }

        # Check if in ML training workflow
        if session.workflow_type == WorkflowType.ML_TRAINING:
            if session.current_state == MLTrainingState.AWAITING_DATA.value:
                # Auto-advance to SELECTING_TARGET
                success, error, missing = await state_manager.transition_state(
                    session, MLTrainingState.SELECTING_TARGET.value
                )

                if success:
                    response = response_builder.build_workflow_prompt(
                        MLTrainingState.SELECTING_TARGET.value, session
                    )
                    await processing_msg.edit_text(response, parse_mode="Markdown")
                    return

        # Generate success message
        success_message = loader.get_data_summary(df, metadata)
        await processing_msg.edit_text(success_message, parse_mode="Markdown")

        logger.info(f"Successfully processed file for user {user_id}: {metadata['shape']}")

    except ValidationError as e:
        error_message = (
            f"❌ **File Validation Error**\n\n"
            f"**Issue:** {e.message}\n\n"
            f"**Supported formats:** CSV, Excel (.xlsx)\n"
            f"**Maximum size:** 10 MB"
        )
        await processing_msg.edit_text(error_message, parse_mode="Markdown")

    except DataError as e:
        error_message = (
            f"❌ **Data Processing Error**\n\n"
            f"**Issue:** {e.message}\n\n"
            f"**Common solutions:**\n"
            f"• Ensure CSV has headers\n"
            f"• Check for corrupted content\n"
            f"• Try different format"
        )
        await processing_msg.edit_text(error_message, parse_mode="Markdown")

    except Exception as e:
        logger.error(f"Unexpected error processing file: {e}", exc_info=True)
        error_response = await integration_layer.format_error(e, session)
        await processing_msg.edit_text(error_response, parse_mode="Markdown")
```

### 3.4 Update Help Messages

```python
MESSAGE_TEMPLATES = {
    "welcome": f"""🤖 Welcome to the Statistical Modeling Agent!

I can help you with:
📊 Statistical analysis of your data
🧠 Machine learning model training
📈 Data predictions and insights

**Quick Start:**
1. Upload a CSV file
2. Use /train for ML workflows
3. Or just ask: "Calculate mean for age"

Type /help for detailed information.""",

    "help": """🆘 Statistical Modeling Agent Help

**Commands:**
/start - Start using the bot
/help - Show this help message
/train - Start ML model training workflow
/predict - Make predictions with trained model
/stats - Quick statistical analysis
/cancel - Cancel active workflow
/diagnostic - Show diagnostic information

**Workflows:**

📊 **Quick Stats** (No workflow needed)
1. Upload CSV file
2. Ask: "Show correlation matrix"
3. Ask: "Calculate mean for age"

🧠 **ML Training** (Multi-step workflow)
1. Upload CSV file
2. /train
3. Select target variable
4. Select feature columns
5. Choose model type
6. Review results

🔮 **ML Prediction** (Multi-step workflow)
1. /predict
2. Select trained model
3. Upload prediction data
4. Get predictions

**Tips:**
• Use /cancel anytime to exit a workflow
• Your data is stored for 30 minutes
• Sessions timeout after inactivity

Need help? Just ask!"""
}
```

---

## Phase 4: Command Handlers

**Duration:** 1-2 weeks
**Dependencies:** Phase 3

### 4.1 WorkflowHandler Class

**File:** `src/bot/workflow_handler.py` (~400 LOC)

See comprehensive implementation in plan above (section on Phase 4).

### 4.2 Command Handler Implementations

#### /train Command

```python
@telegram_handler
@log_user_action("Train command")
async def train_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Start ML training workflow.

    Guides user through:
    1. Data upload (if not done)
    2. Target selection
    3. Feature selection
    4. Model type selection
    5. Training
    """
    user_id = update.effective_user.id
    conversation_id = str(update.effective_chat.id)

    try:
        # Get or create session
        session = await integration_layer.get_or_create_session(
            user_id, conversation_id
        )

        # Check if already in workflow
        if session.workflow_type:
            await update.message.reply_text(
                f"⚠️ **Workflow Already Active**\n\n"
                f"You have an active {session.workflow_type.value} workflow.\n\n"
                f"Use /cancel to reset, or continue with current workflow.",
                parse_mode="Markdown"
            )
            return

        # Check if data uploaded
        if not session.uploaded_data:
            await update.message.reply_text(
                "📁 **No Data Uploaded**\n\n"
                "Please upload your training data (CSV file) first.\n\n"
                "Then use /train to start the training workflow.",
                parse_mode="Markdown"
            )
            return

        # Start workflow
        await state_manager.start_workflow(session, WorkflowType.ML_TRAINING)

        # Transition to SELECTING_TARGET (data already present)
        success, error, missing = await state_manager.transition_state(
            session, MLTrainingState.SELECTING_TARGET.value
        )

        if success:
            prompt = response_builder.build_workflow_prompt(
                MLTrainingState.SELECTING_TARGET.value, session
            )
            await update.message.reply_text(prompt, parse_mode="Markdown")
        else:
            await update.message.reply_text(
                f"❌ **Error Starting Workflow**\n\n{error}",
                parse_mode="Markdown"
            )

    except InvalidStateTransitionError as e:
        await update.message.reply_text(
            f"⚠️ **Workflow Error**\n\n{e.message}\n\nUse /cancel to reset.",
            parse_mode="Markdown"
        )
    except Exception as e:
        logger.error(f"Error in train_handler: {e}", exc_info=True)
        error_response = await integration_layer.format_error(e, session)
        await update.message.reply_text(error_response, parse_mode="Markdown")
```

#### /predict, /stats, /cancel Commands

Similar implementations (see comprehensive plan above).

### 4.3 Register Commands

**File:** `src/bot/telegram_bot.py`

Add after existing handlers:

```python
# Command handlers
app.add_handler(CommandHandler("train", train_handler))
app.add_handler(CommandHandler("predict", predict_handler))
app.add_handler(CommandHandler("stats", stats_handler))
app.add_handler(CommandHandler("cancel", cancel_handler))
```

---

## Phase 5: Configuration & Documentation

**Duration:** Ongoing
**Dependencies:** All phases

### 5.1 Configuration

**File:** `config/config.yaml`

Add section:

```yaml
state_manager:
  session_timeout_minutes: 30
  grace_period_minutes: 5
  max_data_size_mb: 100
  max_history_messages: 50
  cleanup_interval_seconds: 300
  max_concurrent_sessions: 1000

integration_layer:
  enable_workflows: true
  enable_timeout_warnings: true
  auto_cleanup: true
  log_state_transitions: true
```

### 5.2 Documentation

Create/update:
- `dev/implemented/integration-layer.md` (after completion)
- API documentation for new classes
- User guide for workflows
- Troubleshooting guide

---

## Testing Strategy

### Unit Tests (~850 LOC)

1. **IntegrationLayer** (`tests/unit/bot/test_integration_layer.py`)
   - Session creation/retrieval
   - Workflow state checking
   - Message routing logic
   - Error formatting

2. **WorkflowHandler** (`tests/unit/bot/test_workflow_handler.py`)
   - Input parsing (columns, models)
   - Validation logic
   - Prompt generation
   - State transitions

3. **ResponseBuilder** (`tests/unit/bot/test_response_builder.py`)
   - Message formatting
   - Error messages
   - Progress indicators
   - Column selection formatting

### Integration Tests (~1000 LOC)

1. **ML Training Flow** (`tests/integration/test_ml_training_workflow.py`)
   ```python
   async def test_complete_ml_training_workflow():
       # /train → upload → select target → select features → choose model → results
       pass
   ```

2. **ML Prediction Flow** (`tests/integration/test_ml_prediction_workflow.py`)
   ```python
   async def test_complete_ml_prediction_workflow():
       # /predict → select model → upload data → results
       pass
   ```

3. **State Recovery** (`tests/integration/test_state_recovery.py`)
   ```python
   async def test_session_timeout_recovery():
       # Start workflow → timeout → recovery
       pass
   ```

---

## Implementation Schedule

| Week | Phase | Deliverables | Tests |
|------|-------|--------------|-------|
| **1** | Foundation | IntegrationLayer, ResponseBuilder | 500 LOC tests |
| **1-2** | Orchestrator | Refactor to use new StateManager | Update existing tests |
| **2** | Handlers | StateManager integration | 200 LOC tests |
| **2-3** | Commands | /train, /predict, /stats, /cancel | 1,000 LOC tests |
| **3** | Polish | E2E testing, documentation | 150 LOC tests |

---

## Risk Mitigation

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Breaking existing functionality | High | Medium | Incremental changes, regression testing |
| State Manager memory issues | Medium | Low | Aggressive cleanup, monitoring |
| Workflow complexity | Medium | Medium | Clear logging, user-friendly errors |
| Data loss during transitions | Medium | Low | Grace period warnings |
| Concurrent user conflicts | Low | Low | AsyncIO locks already implemented |

---

## Success Criteria

### Functional Requirements
- ✅ Multi-step ML training workflow
- ✅ Multi-step ML prediction workflow
- ✅ Quick stats analysis
- ✅ Workflow cancellation
- ✅ Session timeout handling
- ✅ Data storage across conversation
- ✅ State transition validation
- ✅ Prerequisite checking

### Non-Functional Requirements
- ✅ <100ms session retrieval
- ✅ 1000+ concurrent sessions
- ✅ Thread-safe operations
- ✅ Zero data leakage
- ✅ 100% test pass rate
- ✅ User-friendly errors

### Quality Metrics
- ✅ All tests passing
- ✅ Type annotations complete
- ✅ Documentation comprehensive
- ✅ Code follows patterns
- ✅ Exception handling robust

---

## File Structure

```
src/bot/
  ├── handlers.py                # MODIFIED: StateManager + commands
  ├── integration_layer.py       # NEW: 300 LOC
  ├── workflow_handler.py        # NEW: 400 LOC
  └── response_builder.py        # NEW: 250 LOC

src/core/
  └── orchestrator.py            # MODIFIED: 500 LOC changes

tests/unit/bot/
  ├── test_integration_layer.py  # NEW: 300 LOC
  ├── test_workflow_handler.py   # NEW: 350 LOC
  └── test_response_builder.py   # NEW: 200 LOC

tests/integration/
  ├── test_ml_training_workflow.py   # NEW: 400 LOC
  ├── test_ml_prediction_workflow.py # NEW: 350 LOC
  └── test_state_recovery.py         # NEW: 250 LOC

config/
  └── config.yaml                # MODIFIED

dev/planning/
  └── integration-layer.md       # THIS FILE

dev/implemented/
  └── integration-layer.md       # AFTER COMPLETION
```

**Total Effort:** ~3,300 LOC
- New production: 950 LOC
- New tests: 1,850 LOC
- Modifications: 500 LOC

---

## Next Steps

1. ✅ Planning document created
2. ⏳ User approval received
3. ⏳ Phase 1: IntegrationLayer + ResponseBuilder
4. ⏳ Phase 2: Orchestrator refactoring
5. ⏳ Phase 3: Handler updates
6. ⏳ Phase 4: Command handlers
7. ⏳ Phase 5: Testing & documentation

**Ready to begin implementation!**
