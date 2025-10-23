# Score Workflow: Train + Predict in Single Prompt

**Status**: Planned
**Created**: 2025-01-17
**Author**: Claude Code + User
**Estimated Effort**: 4 days (MVP)

## Overview

The **Score Workflow** is a power-user feature that combines ML training and prediction into a single comprehensive prompt, eliminating the need for multi-step interactions. This workflow is designed for users who:
- Know their dataset structure
- Want to quickly run train + predict pipelines
- Need reproducible workflows (template-based)
- Run regular backtests or experiments

## User Problem

Current workflows (/train and /predict) require 8-13 interaction steps:
- `/train`: Data source → Path → Schema → Target → Features → Model → (Architecture) → Train
- `/predict`: Start → Data source → Path → Schema → Features → Model → Column → Run

**Pain Point**: Repetitive interactions for power users who know exactly what they want.

## Solution

A single-prompt workflow where users provide all configuration upfront in a structured template format.

---

## User Experience Flow

### Step 1: Command Entry
```
User: /score
```

### Step 2: Template Prompt
```
Bot: 🎯 **Score Workflow: Train + Predict**

Provide your configuration in this format:

**Required Fields:**
TRAIN_DATA: /path/to/training_data.csv
TARGET: target_column_name
FEATURES: feature1, feature2, feature3
MODEL: random_forest
PREDICT_DATA: /path/to/prediction_data.csv

**Optional Fields:**
OUTPUT_COLUMN: prediction
HYPERPARAMETERS: {"n_estimators": 100}

**Example:**
TRAIN_DATA: /Users/data/housing_train.csv
TARGET: price
FEATURES: sqft, bedrooms, bathrooms
MODEL: random_forest
PREDICT_DATA: /Users/data/housing_test.csv

**Supported Models:**
Regression: linear, ridge, lasso, random_forest, gradient_boosting
Classification: logistic, decision_tree, random_forest, svm, naive_bayes
Neural Networks: keras_binary_classification, keras_multiclass_classification, keras_regression

Send your template to continue.
```

### Step 3: Template Submission
```
User:
TRAIN_DATA: /Users/data/housing_train.csv
TARGET: price
FEATURES: sqft, bedrooms, bathrooms
MODEL: random_forest
PREDICT_DATA: /Users/data/housing_test.csv
```

### Step 4: Validation & Confirmation
```
Bot: ✅ **Configuration Validated**

📊 **Training Setup:**
• Dataset: housing_train.csv
• Rows: 1,000
• Target: price (continuous)
• Features: 3 columns
• Model: Random Forest
• Task Type: Regression (auto-detected)

🔮 **Prediction Setup:**
• Dataset: housing_test.csv
• Rows: 200
• Features: ✅ All matched
• Output Column: prediction

⏱️ **Estimated Time:** 15-30 seconds

[Confirm] [Edit Template]
```

### Step 5: Execution
```
Bot: 🔄 **Training Model...**
Progress: Loading data → Training → Evaluating

✅ **Training Complete** (8.2s)
• R² Score: 0.85
• MAE: 12,450
• RMSE: 18,230

🔄 **Running Predictions...**
Progress: Loading data → Predicting → Formatting

✅ **Predictions Complete** (1.5s)
• Rows Predicted: 200
• Mean Prediction: $285,432
• Range: $120,000 - $850,000
```

### Step 6: Results
```
Bot: 🎉 **Score Workflow Complete**

📊 **Training Results:**
• Model ID: model_12345_random_forest_20250117
• Performance: R²=0.85, MAE=12,450
• Training Time: 8.2s
• Saved: ✅

🔮 **Prediction Results:**
• Predictions: 200 rows
• Output Column: prediction
• CSV: Attached ⬇️

📥 **Downloads:**
[housing_test_predictions.csv]

💡 **Next Steps:**
• Use model ID for new predictions: /predict
• Modify template and run again: /score
• Train different model: /train

Total Execution Time: 9.7s
```

---

## Technical Architecture

### State Machine Design

**States** (ScoreWorkflowState enum):
```python
class ScoreWorkflowState(Enum):
    AWAITING_TEMPLATE = "awaiting_template"
    VALIDATING_INPUTS = "validating_inputs"
    CONFIRMING_EXECUTION = "confirming_execution"
    TRAINING_MODEL = "training_model"
    RUNNING_PREDICTION = "running_prediction"
    COMPLETE = "complete"
```

**State Transitions:**
```
None → AWAITING_TEMPLATE (on /score command)
↓
AWAITING_TEMPLATE → VALIDATING_INPUTS (template received)
↓
VALIDATING_INPUTS → CONFIRMING_EXECUTION (validation success)
                 → AWAITING_TEMPLATE (validation failed, show errors)
↓
CONFIRMING_EXECUTION → TRAINING_MODEL (user confirms)
                    → AWAITING_TEMPLATE (user clicks back)
↓
TRAINING_MODEL → RUNNING_PREDICTION (training success)
              → COMPLETE (training failed, show error)
↓
RUNNING_PREDICTION → COMPLETE (always)
```

### Data Structures

**ScoreConfig Dataclass:**
```python
@dataclass
class ScoreConfig:
    """Configuration parsed from user template."""
    # Required fields
    train_data_path: str
    target_column: str
    feature_columns: List[str]
    model_type: str
    predict_data_path: str

    # Optional fields
    output_column: str = "prediction"
    hyperparameters: Optional[Dict[str, Any]] = None
    test_split: float = 0.2
    preprocessing: Optional[Dict[str, str]] = None

    # Metadata (populated during validation)
    train_data_shape: Optional[Tuple[int, int]] = None
    predict_data_shape: Optional[Tuple[int, int]] = None
    task_type: Optional[str] = None  # "regression" or "classification"

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of error messages."""
        errors = []
        if not self.train_data_path:
            errors.append("TRAIN_DATA is required")
        if not self.target_column:
            errors.append("TARGET is required")
        if not self.feature_columns:
            errors.append("FEATURES is required")
        if not self.model_type:
            errors.append("MODEL is required")
        if not self.predict_data_path:
            errors.append("PREDICT_DATA is required")
        return errors
```

### Template Parser

**Location**: `src/core/parsers/score_template_parser.py`

**Key Functions:**
```python
def parse_score_template(text: str) -> ScoreConfig:
    """
    Parse user template into ScoreConfig.

    Handles:
    - Case-insensitive keys (TRAIN_DATA, train_data, Train Data)
    - Comma-separated features
    - Optional fields with defaults
    - Whitespace normalization
    """

def validate_score_config(config: ScoreConfig, data_loader: DataLoader) -> Tuple[bool, List[str]]:
    """
    Validate configuration against actual data.

    Steps:
    1. Validate train_data_path exists and accessible
    2. Load train_data schema
    3. Validate target column exists
    4. Validate feature columns exist
    5. Validate model_type is supported
    6. Validate predict_data_path exists
    7. Load predict_data schema
    8. Validate feature compatibility

    Returns:
        (success, error_messages)
    """
```

**Parsing Logic:**
```python
# Example parsing
def parse_line(line: str) -> Tuple[str, str]:
    """Parse 'KEY: value' format."""
    if ':' not in line:
        return None, None

    key, value = line.split(':', 1)
    key = key.strip().lower().replace(' ', '_')
    value = value.strip()

    return key, value

# Feature parsing
def parse_features(value: str, columns: List[str], target: str) -> List[str]:
    """
    Parse feature list with support for:
    - Comma-separated: "age, income, sqft"
    - "all" keyword → all columns except target
    - Whitespace normalization
    """
    if value.lower() == 'all':
        return [col for col in columns if col != target]

    features = [f.strip() for f in value.split(',')]
    return features
```

### Workflow Handler

**Location**: `src/bot/handlers/score_workflow.py`

**Key Functions:**
```python
async def score_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /score command - entry point."""
    # 1. Check for active workflow
    # 2. Start score workflow
    # 3. Send template prompt

async def handle_template_input(update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession):
    """Handle user template submission."""
    # 1. Parse template
    # 2. Validate configuration
    # 3. If valid → show confirmation
    # 4. If invalid → show errors + re-prompt

async def handle_score_confirmation(update: Update, context: ContextTypes.DEFAULT_TYPE, session: UserSession):
    """Handle confirmation button click."""
    # 1. Get config from session
    # 2. Execute workflow
    # 3. Return results

async def execute_score_workflow(config: ScoreConfig, user_id: int, update: Update):
    """
    Execute train + predict workflow.

    Steps:
    1. Load training data
    2. Train model (reuse MLEngine)
    3. Save model
    4. Load prediction data
    5. Run prediction (reuse MLEngine)
    6. Format combined results
    7. Generate CSV output
    """
    # Training phase
    train_df = await data_loader.load_from_local_path(config.train_data_path)

    train_result = ml_engine.train_model(
        data=train_df,
        task_type=config.task_type,  # auto-detected
        model_type=config.model_type,
        target_column=config.target_column,
        feature_columns=config.feature_columns,
        user_id=user_id
    )

    # Prediction phase
    predict_df = await data_loader.load_from_local_path(config.predict_data_path)

    predict_result = ml_engine.predict(
        user_id=user_id,
        model_id=train_result['model_id'],
        data=predict_df
    )

    # Combine results
    return format_score_results(train_result, predict_result, config)
```

### Message Templates

**Location**: `src/bot/messages/score_messages.py`

```python
class ScoreMessages:
    """User-facing messages for score workflow."""

    @staticmethod
    def template_prompt() -> str:
        """Initial template prompt with examples."""

    @staticmethod
    def validation_error(errors: List[str]) -> str:
        """Format validation errors with helpful context."""

    @staticmethod
    def confirmation_summary(config: ScoreConfig, train_shape: Tuple, predict_shape: Tuple) -> str:
        """Show configuration summary before execution."""

    @staticmethod
    def training_progress(progress: float) -> str:
        """Training progress updates."""

    @staticmethod
    def prediction_progress(progress: float) -> str:
        """Prediction progress updates."""

    @staticmethod
    def final_results(train_result: Dict, predict_result: Dict, config: ScoreConfig) -> str:
        """Format combined training + prediction results."""
```

---

## Validation Strategy

### Phase 1: Template Parsing
- **Input**: Raw text from user
- **Output**: ScoreConfig object or list of parse errors
- **Validations**:
  - All required fields present
  - Keys are recognizable (case-insensitive matching)
  - Values are non-empty
  - Format is parseable

### Phase 2: File Validation
- **Reuses**: PathValidator from existing local path training
- **Validations**:
  - Path is absolute
  - Path is within whitelist directories
  - No path traversal attempts
  - File exists and readable
  - File size within limits
  - File extension is allowed (.csv, .xlsx, .parquet)

### Phase 3: Schema Validation
- **Training Data**:
  - Load schema (columns, dtypes)
  - Validate target column exists
  - Validate feature columns exist
  - Auto-detect task type (regression vs classification)
  - Check data quality (non-empty, sufficient rows)

- **Prediction Data**:
  - Load schema
  - Validate all feature columns exist
  - Validate dtypes match training data
  - Allow extra columns (ignored)
  - Check data quality

### Phase 4: Model Validation
- Validate model_type against supported models list
- Check compatibility with task_type
- Validate hyperparameters format (if provided)

### Validation Error Messages

**Missing Field:**
```
❌ Validation Failed

Missing required field: TARGET

Your template must include:
✅ TRAIN_DATA
✅ TARGET  ← Missing
✅ FEATURES
✅ MODEL
✅ PREDICT_DATA

Please add the missing field and try again.
```

**File Not Found:**
```
❌ File Not Found

Cannot access: /path/to/train.csv

Allowed directories:
• /Users/username/datasets
• /home/user/data
• ./data

Please check the file path and try again.
```

**Column Missing:**
```
❌ Column Not Found

Target column 'pric' not found in training data.

Available columns (20):
• price
• sqft
• bedrooms
• bathrooms
...

Did you mean 'price'? (case-sensitive)
```

**Schema Mismatch:**
```
❌ Schema Mismatch

Feature 'sqft' exists in training data but missing in prediction data.

Training columns: price, sqft, bedrooms, bathrooms, age
Prediction columns: price, bedrooms, bathrooms, age

Please ensure both datasets have the same features.
```

---

## Error Handling

### Error Recovery Strategy
All errors return to `AWAITING_TEMPLATE` state with:
1. Clear error message explaining what went wrong
2. Suggestion for how to fix it
3. Previous template preserved for editing (future enhancement)
4. "Try Again" button

### Error Categories

**1. Parse Errors**
- Missing required fields
- Invalid format
- Malformed keys
- **Recovery**: Show template format + examples

**2. File Access Errors**
- File not found
- Permission denied
- Path traversal attempt
- **Recovery**: Show allowed directories + security message

**3. Schema Errors**
- Target column missing
- Feature columns missing
- Dtype mismatch
- **Recovery**: Show available columns + schema comparison

**4. Training Errors**
- Insufficient data
- NaN values
- Model training failure
- **Recovery**: Show detailed error + preprocessing suggestions

**5. Prediction Errors**
- Feature mismatch
- Data loading failure
- Model not found (shouldn't happen)
- **Recovery**: Show expected vs actual features

### Logging Strategy

```python
# Workflow start
logger.info(f"Score workflow started - user={user_id}, timestamp={datetime.now()}")

# Parsing
logger.debug(f"Parsing template - length={len(text)} chars")
logger.info(f"Template parsed - train={config.train_data_path}, model={config.model_type}")

# Validation
logger.info(f"Validating train_data - path={path}")
logger.debug(f"Train data schema - shape={shape}, columns={columns}")

# Training
logger.info(f"Training started - model={model_type}, target={target}, features={len(features)}")
logger.info(f"Training complete - R²={r2:.3f}, time={elapsed:.2f}s, model_id={model_id}")

# Prediction
logger.info(f"Prediction started - model={model_id}, rows={n_rows}")
logger.info(f"Prediction complete - rows={n_rows}, time={elapsed:.2f}s")

# Completion
logger.info(f"Score workflow complete - user={user_id}, total_time={total_time:.2f}s")

# Errors
logger.error(f"Validation failed - errors={errors}")
logger.error(f"Training failed - error={str(e)}", exc_info=True)
```

---

## Integration Points

### 1. Telegram Bot Registration
**File**: `src/bot/telegram_bot.py`

```python
# Add imports
from src.bot.handlers.score_workflow import (
    score_command_handler,
    handle_score_callback
)

# Register command handler
application.add_handler(CommandHandler("score", score_command_handler))

# Register callback handlers
application.add_handler(CallbackQueryHandler(
    handle_score_callback,
    pattern="^score_"
))
```

### 2. State Manager Updates
**File**: `src/core/state_manager.py`

```python
# Add to WorkflowType enum
class WorkflowType(Enum):
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    SCORE_WORKFLOW = "score_workflow"  # NEW

# Add ScoreWorkflowState enum
class ScoreWorkflowState(Enum):
    AWAITING_TEMPLATE = "awaiting_template"
    VALIDATING_INPUTS = "validating_inputs"
    CONFIRMING_EXECUTION = "confirming_execution"
    TRAINING_MODEL = "training_model"
    RUNNING_PREDICTION = "running_prediction"
    COMPLETE = "complete"

# Add to WORKFLOW_TRANSITIONS
SCORE_WORKFLOW_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    None: {ScoreWorkflowState.AWAITING_TEMPLATE.value},
    ScoreWorkflowState.AWAITING_TEMPLATE.value: {
        ScoreWorkflowState.VALIDATING_INPUTS.value
    },
    ScoreWorkflowState.VALIDATING_INPUTS.value: {
        ScoreWorkflowState.CONFIRMING_EXECUTION.value,
        ScoreWorkflowState.AWAITING_TEMPLATE.value
    },
    ScoreWorkflowState.CONFIRMING_EXECUTION.value: {
        ScoreWorkflowState.TRAINING_MODEL.value,
        ScoreWorkflowState.AWAITING_TEMPLATE.value
    },
    ScoreWorkflowState.TRAINING_MODEL.value: {
        ScoreWorkflowState.RUNNING_PREDICTION.value,
        ScoreWorkflowState.COMPLETE.value
    },
    ScoreWorkflowState.RUNNING_PREDICTION.value: {
        ScoreWorkflowState.COMPLETE.value
    },
    ScoreWorkflowState.COMPLETE.value: set()
}

WORKFLOW_TRANSITIONS[WorkflowType.SCORE_WORKFLOW] = SCORE_WORKFLOW_TRANSITIONS
```

### 3. Help Text Update
**File**: `src/bot/handlers.py`

```python
MESSAGE_TEMPLATES["help"] = """
🆘 Statistical Modeling Agent Help

Commands:
/start - Start using the bot
/help - Show this help message
/train - Start ML training workflow (interactive, step-by-step)
/predict - Make predictions with trained model (interactive)
/score - Train + Predict in one step (advanced, template-based)  # NEW
/cancel - Cancel current workflow

How to use:
1. Upload Data: Send a CSV file (for /train)
2. Request Analysis: Choose your workflow
   • /train - Guided step-by-step training
   • /predict - Guided step-by-step prediction
   • /score - Single-prompt train+predict (power users)
3. Get Results: Download predictions and model info

Workflows:
📊 /train - Interactive ML training (beginners)
🔮 /predict - Interactive prediction (existing models)
⚡ /score - Combined train+predict (power users, fast)

...
"""
```

### 4. Reused Components

**No modifications needed** for these components:
- `src/engines/ml_engine.py` - `train_model()`, `predict()`
- `src/utils/path_validator.py` - `validate_file_path()`
- `src/processors/data_loader.py` - `load_from_local_path()`
- `src/core/state_manager.py` - Session management (only additions)
- `src/processors/result_processor.py` - Results formatting

---

## Security Considerations

### Path Validation
- **Reuse**: PathValidator with whitelist enforcement
- **Checks**: Path traversal, symlink resolution, whitelist directories
- **No changes needed**: Existing security infrastructure is sufficient

### Input Sanitization
```python
# Column name validation
COLUMN_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_]+$')

def sanitize_column_name(name: str) -> str:
    """Validate column name is safe."""
    if not COLUMN_NAME_PATTERN.match(name):
        raise ValidationError(f"Invalid column name: {name}")
    return name
```

### Model Type Validation
```python
SUPPORTED_MODELS = {
    'linear', 'ridge', 'lasso', 'elasticnet', 'polynomial',
    'logistic', 'decision_tree', 'random_forest', 'gradient_boosting',
    'svm', 'naive_bayes',
    'keras_binary_classification', 'keras_multiclass_classification',
    'keras_regression'
}

def validate_model_type(model_type: str) -> str:
    """Validate model type against whitelist."""
    if model_type not in SUPPORTED_MODELS:
        raise ValidationError(f"Unsupported model: {model_type}")
    return model_type
```

### Resource Limits
- File size limits: Existing in PathValidator (configurable)
- Memory limits: Set during training (existing in MLEngine)
- Execution timeout: 5 minutes max for full workflow
- Concurrent sessions: Existing StateManager limits

---

## Performance Optimization

### Async Operations
```python
# Parallel file loading (future enhancement)
train_df, predict_df = await asyncio.gather(
    data_loader.load_from_local_path(config.train_data_path),
    data_loader.load_from_local_path(config.predict_data_path)
)
```

### Memory Management
```python
# Clear training data after model training
train_df = None  # Allow garbage collection

# Load prediction data only after training complete
predict_df = await data_loader.load_from_local_path(config.predict_data_path)
```

### Progress Updates
```python
# Training progress
await update.message.reply_text("🔄 Training model... (loading data)")
await update.message.reply_text("🔄 Training model... (fitting)")
await update.message.reply_text("🔄 Training model... (evaluating)")

# Prediction progress
await update.message.reply_text("🔄 Running predictions... (loading data)")
await update.message.reply_text("🔄 Running predictions... (predicting)")
```

### Expected Performance
- Template parsing: <100ms
- File validation: <500ms
- Schema validation: <1s
- Training: 5-30s (dataset dependent)
- Prediction: <5s
- **Total**: <40s typical, <60s maximum

---

## Testing Strategy

### Unit Tests
**File**: `tests/unit/test_score_template_parser.py`

```python
class TestScoreTemplateParser:
    """Test template parsing logic."""

    def test_parse_valid_template(self):
        """Test parsing of valid template with all fields."""

    def test_parse_missing_required_field(self):
        """Test error handling for missing required fields."""

    def test_parse_case_insensitive_keys(self):
        """Test that keys are case-insensitive."""

    def test_parse_feature_list(self):
        """Test comma-separated feature parsing."""

    def test_parse_features_all_keyword(self):
        """Test 'all' keyword for features."""

    def test_parse_optional_fields(self):
        """Test parsing of optional fields."""

    def test_parse_malformed_template(self):
        """Test error handling for malformed input."""

    def test_validate_score_config(self):
        """Test configuration validation logic."""
```

### Integration Tests
**File**: `tests/integration/test_score_workflow.py`

```python
class TestScoreWorkflow:
    """Test complete score workflow end-to-end."""

    @pytest.fixture
    def mock_bot(self):
        """Mock Telegram bot for testing."""

    @pytest.fixture
    def sample_train_data(self):
        """Sample training CSV file."""

    @pytest.fixture
    def sample_predict_data(self):
        """Sample prediction CSV file."""

    async def test_complete_workflow_success(self):
        """Test full workflow from template to results."""

    async def test_workflow_missing_file(self):
        """Test error handling when file not found."""

    async def test_workflow_invalid_column(self):
        """Test error handling for invalid target column."""

    async def test_workflow_schema_mismatch(self):
        """Test error handling for schema mismatches."""

    async def test_workflow_training_failure(self):
        """Test error handling when training fails."""

    async def test_workflow_back_button(self):
        """Test back button navigation."""

    async def test_workflow_cancel(self):
        """Test workflow cancellation."""

    async def test_multiple_users_concurrent(self):
        """Test multiple users running workflows simultaneously."""
```

### Test Coverage Goals
- **Template Parser**: >95%
- **Score Workflow Handler**: >90%
- **Message Templates**: >80%
- **Overall**: >90%

### Test Data
```python
# Valid template
VALID_TEMPLATE = """
TRAIN_DATA: /test/data/train.csv
TARGET: price
FEATURES: sqft, bedrooms, bathrooms
MODEL: random_forest
PREDICT_DATA: /test/data/test.csv
"""

# Invalid templates for error testing
MISSING_TARGET = """
TRAIN_DATA: /test/data/train.csv
FEATURES: sqft, bedrooms
MODEL: random_forest
PREDICT_DATA: /test/data/test.csv
"""

INVALID_MODEL = """
TRAIN_DATA: /test/data/train.csv
TARGET: price
FEATURES: sqft, bedrooms
MODEL: invalid_model_name
PREDICT_DATA: /test/data/test.csv
"""
```

---

## Deployment Strategy

### Phase 1: Development (2-3 days)
- Implement template parser
- Implement workflow handler
- Implement message templates
- Add state manager changes
- Local testing

### Phase 2: Testing (1 day)
- Write unit tests (>90% coverage)
- Write integration tests
- Manual testing with real Telegram bot
- Edge case testing
- Performance testing

### Phase 3: Documentation (0.5 day)
- Update CLAUDE.md
- Update README.md
- Create user guide with examples
- Document API for developers

### Phase 4: Staging (0.5 day)
- Deploy to staging environment
- Test with staging Telegram bot
- Beta user testing
- Collect feedback

### Phase 5: Production (0.5 day)
- Feature flag for gradual rollout
- Monitor metrics
- Address any issues
- Full rollout

### Rollback Plan
- Feature is additive only (no modifications to existing workflows)
- Can disable with feature flag if critical issues found
- No database migrations needed
- No breaking changes to existing code

### Monitoring After Release
```python
# Metrics to track
- score_workflow_started_total
- score_workflow_completed_total
- score_workflow_failed_total
- score_workflow_duration_seconds
- score_validation_errors_total
- score_training_errors_total
- score_prediction_errors_total

# Log analysis
- Most common validation errors
- Average execution time
- User adoption rate
- Error rates by error type
```

---

## Future Enhancements (Post-MVP)

### Phase 2: Enhanced Features
- Hyperparameters support
- Preprocessing options (scaling, missing value handling)
- Custom model naming
- Multiple output formats (CSV, Excel, Parquet)
- Progress bars with ETA

### Phase 3: Power User Features
- Template library (save/load common templates)
- Single-line quick format: `QUICK: train.csv -> test.csv TARGET=price MODEL=auto`
- Dry run mode: `DRY_RUN: true` (validates without executing)
- Batch processing: Process multiple prediction datasets at once

### Phase 4: Advanced Analytics
- Model comparison mode: Train multiple models, compare results
- Feature importance visualization
- Confusion matrix for classification
- Residual plots for regression
- Auto-optimization: Find best hyperparameters

---

## Comparison with Existing Workflows

### Feature Matrix

| Feature | /train | /predict | /score |
|---------|--------|----------|--------|
| Interaction Steps | 5-8 | 6-8 | 1 |
| User Type | Beginner | Intermediate | Advanced |
| Reproducibility | Low | Medium | High |
| Speed | Slow | Slow | Fast |
| Guided | ✅ | ✅ | ❌ |
| Template-based | ❌ | ❌ | ✅ |
| Combined Train+Predict | ❌ | ❌ | ✅ |
| Back Navigation | ✅ | ✅ | ✅ |
| Error Recovery | Inline | Inline | Full Restart |

### When to Use Each Workflow

**Use /train**:
- First time using the bot
- Exploring new datasets
- Learning ML concepts
- Want step-by-step guidance
- Unsure about model selection

**Use /predict**:
- Have existing trained model
- Running predictions on new data
- Want guided feature selection
- Need to confirm prediction column

**Use /score**:
- Know your dataset structure
- Running regular backtests
- Production workflows
- Automated experiments
- Reproducible pipelines
- Speed is important

---

## Success Criteria

### Functional Requirements
✅ User can submit template in single prompt
✅ All required fields validated
✅ File paths validated with security checks
✅ Schema validation catches mismatches
✅ Training executes successfully
✅ Prediction executes successfully
✅ Combined results displayed clearly
✅ CSV download provided
✅ Back button navigation works
✅ Error messages are clear and actionable

### Non-Functional Requirements
✅ <5% error rate on valid templates
✅ <40s typical execution time
✅ >90% test coverage
✅ Zero regressions in /train or /predict
✅ Clear documentation for users
✅ Monitoring in place

### User Satisfaction
✅ Power users prefer /score over /train + /predict
✅ Templates are easy to create and modify
✅ Error messages help users fix issues
✅ Execution speed meets expectations
✅ Results format is clear and useful

---

## Risk Assessment

### Technical Risks

**Risk**: Template parsing errors (Medium)
- **Mitigation**: Extensive unit tests, fuzzing, user testing
- **Impact**: Low (users can retry with fixes)

**Risk**: Schema validation complexity (Medium)
- **Mitigation**: Reuse existing validators, comprehensive error messages
- **Impact**: Medium (could block valid workflows)

**Risk**: Performance issues with large datasets (Low)
- **Mitigation**: Progress updates, async operations, memory management
- **Impact**: Low (users expect some wait time)

**Risk**: Security vulnerabilities (Low)
- **Mitigation**: Reuse PathValidator, input sanitization, whitelist validation
- **Impact**: High (if exploited)

### User Experience Risks

**Risk**: Confusing error messages (Medium)
- **Mitigation**: User testing, clear examples, inline suggestions
- **Impact**: Medium (users abandon workflow)

**Risk**: Steep learning curve (Medium)
- **Mitigation**: Comprehensive examples, clear documentation, /train fallback
- **Impact**: Low (target is power users)

**Risk**: Template format too rigid (Low)
- **Mitigation**: Support flexible parsing, case-insensitive, whitespace tolerant
- **Impact**: Low (users adapt quickly)

### Operational Risks

**Risk**: Increased error rates (Low)
- **Mitigation**: Comprehensive validation, clear error recovery
- **Impact**: Medium (support burden)

**Risk**: Performance degradation (Low)
- **Mitigation**: Resource limits, monitoring, async operations
- **Impact**: Medium (system slowdown)

**Risk**: Bugs in production (Medium)
- **Mitigation**: Thorough testing, staged rollout, feature flag
- **Impact**: Medium (user frustration)

---

## Estimated Effort Breakdown

### Development Tasks
- Template parser implementation: **8 hours**
- Workflow handler implementation: **12 hours**
- Message templates: **4 hours**
- State manager updates: **2 hours**
- Integration with bot: **2 hours**
- **Total Development**: **28 hours (3.5 days)**

### Testing Tasks
- Unit tests: **6 hours**
- Integration tests: **6 hours**
- Manual testing: **4 hours**
- **Total Testing**: **16 hours (2 days)**

### Documentation Tasks
- CLAUDE.md updates: **2 hours**
- User guide: **2 hours**
- This specification: **2 hours**
- **Total Documentation**: **6 hours (0.75 days)**

### Deployment Tasks
- Staging deployment: **2 hours**
- Beta testing: **4 hours**
- Production rollout: **2 hours**
- **Total Deployment**: **8 hours (1 day)**

### **Grand Total**: **58 hours (7.25 days)**

**MVP Only**: **44 hours (5.5 days)** (excluding Phase 2+ enhancements)

---

## Conclusion

The Score Workflow fills a critical gap for power users who need fast, reproducible, template-based train+predict pipelines. By consolidating 13-16 interaction steps into a single comprehensive prompt, we can significantly improve productivity for advanced users while maintaining safety through comprehensive validation.

The implementation is well-scoped, leverages existing infrastructure (MLEngine, PathValidator, DataLoader), and introduces minimal risk to existing workflows. With clear error messages, back button navigation, and thorough testing, we can deliver a high-quality MVP in ~5 days.

**Next Steps**:
1. Review and approve this specification
2. Begin Phase 1: Template Parser implementation
3. Iterative development with testing
4. Beta release for feedback
5. Production rollout with monitoring

---

## Appendix: Template Examples

### Example 1: Simple Regression
```
TRAIN_DATA: /Users/data/housing_train.csv
TARGET: price
FEATURES: sqft, bedrooms, bathrooms
MODEL: random_forest
PREDICT_DATA: /Users/data/housing_test.csv
```

### Example 2: Classification with All Features
```
TRAIN_DATA: /data/credit_train.csv
TARGET: default
FEATURES: all
MODEL: logistic
PREDICT_DATA: /data/credit_test.csv
OUTPUT_COLUMN: default_prediction
```

### Example 3: Neural Network (Future)
```
TRAIN_DATA: /data/images_train.csv
TARGET: category
FEATURES: all
MODEL: keras_multiclass_classification
HYPERPARAMETERS: {"epochs": 200, "batch_size": 64}
PREDICT_DATA: /data/images_test.csv
```

### Example 4: Advanced Preprocessing (Future)
```
TRAIN_DATA: /data/sales_train.csv
TARGET: revenue
FEATURES: marketing, season, category, region
MODEL: gradient_boosting
PREPROCESSING: {"scaling": "standard", "missing": "mean"}
HYPERPARAMETERS: {"n_estimators": 200, "learning_rate": 0.05}
PREDICT_DATA: /data/sales_forecast.csv
SAVE_MODEL_AS: sales_forecaster_2025
OUTPUT_FORMAT: excel
```

---

**Document Version**: 1.0
**Last Updated**: 2025-01-17
**Status**: Approved for Implementation