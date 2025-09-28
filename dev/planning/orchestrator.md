# Enhanced Orchestrator Implementation Plan

## Overview

The current `src/core/orchestrator.py` serves as a basic task router that directs parsed requests to the appropriate engines. However, to fulfill the complete orchestrator requirements, we need to enhance it into a comprehensive task coordination system with state management, workflow orchestration, and intelligent error handling.

## Current State Analysis

### ✅ What We Have
- **Request Routing**: Routes TaskDefinition objects to StatsEngine
- **Basic Error Handling**: Timeout and exception management
- **execute_task() Method**: Async task execution with standardized results
- **Engine Integration**: Works with existing StatsEngine

### ❌ Missing Components
- **Conversation State Management**: No state tracking across user sessions
- **DataLoader Integration**: No coordination with data loading pipeline
- **Multi-Step Workflows**: No support for complex workflows like ML training
- **Error Recovery**: No retry strategies or user feedback mechanisms
- **Progress Tracking**: No long-operation progress indicators

## Implementation Plan

### Phase 1: State Management Architecture

**Objective**: Add conversation state tracking and workflow persistence

**Files to Modify**:
- `src/core/orchestrator.py`

**Components to Add**:

#### ConversationState Class
```python
@dataclass
class ConversationState:
    user_id: int
    conversation_id: str
    workflow_state: WorkflowState
    current_step: Optional[str]
    context: Dict[str, Any]
    partial_results: Dict[str, Any]
    data_sources: List[str]
    created_at: datetime
    last_activity: datetime
```

#### WorkflowState Enum
```python
class WorkflowState(Enum):
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
```

#### StateManager Class
```python
class StateManager:
    def __init__(self, ttl_minutes: int = 60):
        self.states: Dict[str, ConversationState] = {}
        self.ttl = ttl_minutes

    async def get_state(self, user_id: int, conversation_id: str) -> ConversationState
    async def save_state(self, state: ConversationState) -> None
    async def clear_state(self, user_id: int, conversation_id: str) -> None
    async def cleanup_expired(self) -> None
```

### Phase 2: DataLoader Integration

**Objective**: Coordinate data lifecycle management with the existing DataLoader

**Files to Modify**:
- `src/core/orchestrator.py`
- `src/bot/handlers.py`

**Components to Add**:

#### DataManager Class
```python
class DataManager:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader
        self.data_cache: Dict[str, Tuple[pd.DataFrame, Dict[str, Any]]] = {}

    async def load_data(self, telegram_file, file_name: str, user_id: int) -> str
    async def get_data(self, data_id: str) -> Tuple[pd.DataFrame, Dict[str, Any]]
    async def validate_data(self, data_id: str, requirements: Dict[str, Any]) -> bool
    async def cache_data(self, data_id: str, dataframe: pd.DataFrame, metadata: Dict[str, Any]) -> None
    async def clear_cache(self, user_id: int) -> None
```

**Integration Points**:
- Import `DataLoader` from `src/processors/data_loader.py`
- Handle data validation pipeline before engine execution
- Support multiple data sources per user session
- Implement data caching with user-specific namespacing

### Phase 3: Multi-Step Workflow Engine

**Objective**: Implement state machines for complex multi-step processes

**Files to Modify**:
- `src/core/orchestrator.py`

**Components to Add**:

#### WorkflowEngine Class
```python
class WorkflowEngine:
    def __init__(self, state_manager: StateManager):
        self.state_manager = state_manager
        self.workflow_templates = self._init_templates()

    async def start_workflow(self, workflow_type: str, state: ConversationState) -> Dict[str, Any]
    async def advance_workflow(self, state: ConversationState, user_input: str) -> Dict[str, Any]
    async def validate_transition(self, from_state: WorkflowState, to_state: WorkflowState) -> bool
    async def get_next_prompt(self, state: ConversationState) -> str
    async def cancel_workflow(self, state: ConversationState) -> None
```

#### Workflow Templates
```python
WORKFLOW_TEMPLATES = {
    "ml_training": {
        "steps": [
            {"state": "AWAITING_DATA", "prompt": "Please upload your training data"},
            {"state": "SELECTING_TARGET", "prompt": "Select target variable from: {columns}"},
            {"state": "SELECTING_FEATURES", "prompt": "Select feature columns from: {features}"},
            {"state": "CONFIGURING_MODEL", "prompt": "Choose model type: neural_network, random_forest, linear_regression"},
            {"state": "TRAINING", "prompt": "Training model... Please wait"},
            {"state": "COMPLETED", "prompt": "Training complete! Results: {results}"}
        ],
        "validation": {...},
        "error_recovery": {...}
    },
    "stats_analysis": {
        "steps": [
            {"state": "AWAITING_DATA", "prompt": "Please upload your data for analysis"},
            {"state": "DATA_LOADED", "prompt": "Data loaded. What analysis would you like?"},
            {"state": "COMPLETED", "prompt": "Analysis complete: {results}"}
        ]
    }
}
```

### Phase 4: Error Recovery & Feedback System

**Objective**: Implement intelligent error handling with user feedback

**Files to Modify**:
- `src/core/orchestrator.py`

**Components to Add**:

#### ErrorRecoverySystem Class
```python
class ErrorRecoverySystem:
    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        self.retry_strategies = self._init_strategies()

    async def handle_error(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]
    async def retry_with_backoff(self, operation: Callable, max_attempts: int) -> Any
    async def suggest_recovery(self, error_type: str, context: Dict[str, Any]) -> str
    async def escalate_to_user(self, error: Exception, suggestions: List[str]) -> str
```

#### FeedbackLoop Class
```python
class FeedbackLoop:
    def __init__(self, formatter):
        self.formatter = formatter

    async def request_clarification(self, ambiguous_input: str, options: List[str]) -> str
    async def show_progress(self, operation: str, progress: float) -> str
    async def confirm_action(self, action: str, consequences: List[str]) -> str
    async def suggest_alternatives(self, failed_action: str, alternatives: List[str]) -> str
```

**Error Recovery Strategies**:
- **Transient Failures**: Exponential backoff retry (network, temporary file issues)
- **Data Issues**: Suggest data cleaning or alternative columns
- **Parsing Failures**: Provide example requests and column suggestions
- **Engine Failures**: Fallback to simpler analysis or manual mode

### Phase 5: Enhanced execute_task() Method

**Objective**: Enhance the existing execute_task method with state awareness

**Files to Modify**:
- `src/core/orchestrator.py`

**Enhancements**:

```python
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
    # Get or create conversation state
    state = await self.state_manager.get_state(task.user_id, task.conversation_id)

    # Check if this is part of a multi-step workflow
    if state.workflow_state != WorkflowState.IDLE:
        return await self.workflow_engine.advance_workflow(state, task)

    # Handle data loading if needed
    if data is None and state.data_sources:
        data, metadata = await self.data_manager.get_data(state.data_sources[-1])

    # Execute with enhanced error recovery
    try:
        result = await self._execute_with_recovery(task, data, timeout, progress_callback)

        # Update state with results
        state.partial_results[task.operation] = result
        await self.state_manager.save_state(state)

        return result

    except Exception as e:
        return await self.error_recovery.handle_error(e, {
            "task": task,
            "state": state,
            "data_shape": data.shape if data is not None else None
        })
```

### Phase 6: Testing & Documentation

**Files to Create**:

#### Comprehensive Test Suite
- `tests/unit/test_orchestrator_enhanced.py`
  - State management tests
  - Workflow engine tests
  - Error recovery tests
  - DataLoader integration tests
  - Multi-step workflow scenarios

#### Test Scenarios
```python
class TestEnhancedOrchestrator:
    def test_ml_training_workflow(self):
        """Test complete ML training workflow from data upload to results"""

    def test_conversation_state_persistence(self):
        """Test state is maintained across multiple requests"""

    def test_error_recovery_strategies(self):
        """Test various error scenarios and recovery mechanisms"""

    def test_data_loader_integration(self):
        """Test seamless data loading and caching"""

    def test_concurrent_user_sessions(self):
        """Test multiple users with separate workflow states"""
```

## Architecture Overview

### Component Relationships

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User Request  │───▶│ Enhanced         │───▶│  State Manager  │
│   (Telegram)    │    │ Orchestrator     │    │   (In-Memory)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Manager  │◀───│  Workflow Engine │───▶│ Error Recovery  │
│ (DataLoader)    │    │ (State Machine)  │    │   System        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Stats Engine   │◀───│   Task Routing   │───▶│ Feedback Loop   │
│  (Existing)     │    │   (Enhanced)     │    │ (User Clarity)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Data Flow

1. **Request Received**: User sends message via Telegram
2. **State Check**: Orchestrator checks for existing conversation state
3. **Workflow Route**: Determines if this is a new task or workflow continuation
4. **Data Coordination**: Loads/validates data through DataManager if needed
5. **Execution**: Routes to appropriate engine with error recovery
6. **State Update**: Updates conversation state with results
7. **Response**: Formats and returns result with next step guidance

## Integration Points

### Existing System Integration
- **Parser → Orchestrator**: Enhanced to pass workflow context
- **Orchestrator → Engines**: Existing integration maintained
- **ResultFormatter**: Enhanced to include workflow status

### New System Integration
- **DataLoader → Orchestrator**: Direct integration for data lifecycle
- **StateManager → Orchestrator**: Persistent workflow state
- **ErrorHandler → Orchestrator**: Intelligent error recovery
- **FeedbackLoop → User**: Enhanced user interaction

## Success Metrics

### Functional Requirements
- [x] Routes parsed requests to appropriate engines ✅ (Enhanced)
- [x] Manages conversation state for multi-step workflows ✅ (New)
- [x] Coordinates between data_loader, parser, and engines ✅ (New)
- [x] Handles error recovery and user feedback ✅ (New)
- [x] Implements enhanced execute_task() method ✅ (Enhanced)

### Quality Standards
- **State Persistence**: 99.9% reliability across user sessions
- **Workflow Completion**: >95% success rate for multi-step processes
- **Error Recovery**: <5s average recovery time for common failures
- **User Experience**: Clear feedback at each workflow step
- **Performance**: <1s state operations, <30s complex workflows

## Implementation Timeline

### Phase 1-2: Foundation (Week 1)
- State management classes
- DataLoader integration
- Basic workflow structure

### Phase 3-4: Workflow & Recovery (Week 2)
- Multi-step workflow engine
- Error recovery system
- User feedback mechanisms

### Phase 5-6: Enhancement & Testing (Week 3)
- Enhanced execute_task() method
- Comprehensive test suite
- Documentation completion

## Benefits

### User Experience
- **Guided Workflows**: Step-by-step guidance for complex operations
- **Session Continuity**: Workflows persist across message sessions
- **Smart Error Recovery**: Helpful suggestions when things go wrong
- **Progress Visibility**: Clear indication of workflow status

### System Reliability
- **Robust Error Handling**: Multiple recovery strategies
- **State Consistency**: Reliable workflow state management
- **Data Integrity**: Coordinated data loading and validation
- **Scalable Architecture**: Support for concurrent user sessions

### Developer Experience
- **Extensible Workflows**: Easy to add new multi-step processes
- **Comprehensive Testing**: Full coverage of workflow scenarios
- **Clear Architecture**: Well-defined component responsibilities
- **Maintainable Code**: Modular design with clear interfaces

This enhanced orchestrator will transform the Statistical Modeling Agent from a basic task router into a sophisticated workflow coordination system capable of handling complex, multi-step statistical and machine learning processes with robust error handling and excellent user experience.