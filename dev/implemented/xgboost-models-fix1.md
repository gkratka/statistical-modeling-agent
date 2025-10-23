# XGBoost Parameter Configuration Implementation Plan

**Status**: Planning
**Created**: 2025-10-19
**Priority**: High

## Overview

### Current Issue
When users select "Gradient Boosting" in the classification model menu, the bot immediately trains a sklearn `GradientBoostingClassifier` with default parameters. There is no option to:
- Choose XGBoost instead of sklearn
- Configure XGBoost hyperparameters (similar to Keras workflow)
- Differentiate between the two gradient boosting implementations

### Proposed Solution
Implement a parameter configuration workflow for XGBoost models that:
1. Distinguishes between sklearn Gradient Boosting and XGBoost
2. Provides interactive parameter selection (similar to Keras models)
3. Allows users to customize key hyperparameters:
   - `n_estimators` - Number of boosting rounds
   - `max_depth` - Maximum tree depth
   - `learning_rate` - Step size shrinkage
   - `subsample` - Subsample ratio of training instances
   - `colsample_bytree` - Subsample ratio of columns

## UI/UX Design

### User Workflow

**Current (Broken)**:
```
Classification Models → Gradient Boosting (click) → Immediate sklearn training
```

**Proposed**:
```
Classification Models → Gradient Boosting (sklearn) | XGBoost Classification
                                ↓ (if XGBoost clicked)
                        Step 1/5: n_estimators
                                ↓
                        Step 2/5: max_depth
                                ↓
                        Step 3/5: learning_rate
                                ↓
                        Step 4/5: subsample
                                ↓
                        Step 5/5: colsample_bytree
                                ↓
                        Training with custom parameters
```

### Button Labels

**Classification Models Menu**:
- `Logistic Regression` → `logistic`
- `Decision Tree` → `decision_tree`
- `Random Forest` → `random_forest`
- **`Gradient Boosting (sklearn)`** → `gradient_boosting` ✅ NEW LABEL
- **`XGBoost Classification`** → `xgboost_binary_classification` ✅ NEW OPTION
- `Support Vector Machine` → `svm`
- `Naive Bayes` → `naive_bayes`

**Regression Models Menu**:
- `Linear Regression` → `linear`
- `Ridge Regression (L2)` → `ridge`
- `Lasso Regression (L1)` → `lasso`
- `ElasticNet (L1+L2)` → `elasticnet`
- `Polynomial Regression` → `polynomial`
- **`XGBoost Regression`** → `xgboost_regression` ✅ ALREADY EXISTS

### Parameter Selection Screens

#### Step 1/5: n_estimators
```
🧠 XGBoost Configuration

Step 1/5: Number of Boosting Rounds

How many trees to build?
(More trees = better fit, but slower training)

[50 trees]
[100 trees (recommended)]
[200 trees]
[Custom]
[← Back]
```

#### Step 2/5: max_depth
```
🧠 XGBoost Configuration

Step 2/5: Maximum Tree Depth

How deep should each tree be?
(Deeper = more complex patterns, risk of overfitting)

[3 levels]
[6 levels (recommended)]
[9 levels]
[Custom]
[← Back]
```

#### Step 3/5: learning_rate
```
🧠 XGBoost Configuration

Step 3/5: Learning Rate

How fast should the model learn?
(Lower = more stable, but needs more trees)

[0.01 (conservative)]
[0.1 (recommended)]
[0.3 (aggressive)]
[Custom]
[← Back]
```

#### Step 4/5: subsample
```
🧠 XGBoost Configuration

Step 4/5: Subsample Ratio

What fraction of data to use per tree?
(Lower = more diverse trees, prevents overfitting)

[0.6 (60%)]
[0.8 (80% - recommended)]
[1.0 (100% - all data)]
[Custom]
[← Back]
```

#### Step 5/5: colsample_bytree
```
🧠 XGBoost Configuration

Step 5/5: Column Subsample Ratio

What fraction of features to use per tree?
(Lower = more diverse trees, prevents overfitting)

[0.6 (60%)]
[0.8 (80% - recommended)]
[1.0 (100% - all features)]
[Custom]
[← Back]
```

## Technical Implementation

### State Management

**New Session State**:
```python
session.selections['xgboost_config'] = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

**No new workflow state needed** - reuse existing Keras pattern

### Code Structure

#### 1. Update Model Selection Menu

**File**: `src/bot/ml_handlers/ml_training_local_path.py`
**Location**: Lines 920-946 (handle_model_category_selection)

**Current**:
```python
"classification": [
    ("Logistic Regression", "logistic"),
    ("Decision Tree", "decision_tree"),
    ("Random Forest", "random_forest"),
    ("Gradient Boosting", "gradient_boosting"),  # ❌ Ambiguous
    ("Support Vector Machine", "svm"),
    ("Naive Bayes", "naive_bayes"),
    ("XGBoost Classification", "xgboost_binary_classification")
],
```

**Updated**:
```python
"classification": [
    ("Logistic Regression", "logistic"),
    ("Decision Tree", "decision_tree"),
    ("Random Forest", "random_forest"),
    ("Gradient Boosting (sklearn)", "gradient_boosting"),  # ✅ Clear label
    ("XGBoost Classification", "xgboost_binary_classification"),  # ✅ Distinct option
    ("Support Vector Machine", "svm"),
    ("Naive Bayes", "naive_bayes")
],
```

#### 2. Update Model Selection Handler

**File**: `src/bot/ml_handlers/ml_training_local_path.py`
**Location**: Line 1023 (handle_model_selection)

**Add XGBoost Detection**:
```python
# Check if this is a Keras model - needs parameter configuration
if model_type.startswith('keras_'):
    print(f"🧠 DEBUG: Keras model selected, starting parameter configuration")
    await self._start_keras_config(query, session)
# NEW: Check if XGBoost model - needs parameter configuration
elif model_type.startswith('xgboost_'):
    print(f"🚀 DEBUG: XGBoost model selected, starting parameter configuration")
    await self._start_xgboost_config(query, session, model_type)
else:
    # sklearn models - start training immediately
    # ...existing code...
```

#### 3. Implement XGBoost Parameter Handlers

**File**: `src/bot/ml_handlers/ml_training_local_path.py`
**Location**: After `_start_keras_config` method

**New Methods** (5 parameter handlers + 1 starter):

```python
async def _start_xgboost_config(
    self,
    query,
    session,
    model_type: str
) -> None:
    """Start XGBoost parameter configuration workflow."""
    # Initialize XGBoost config dict with defaults from template
    from src.engines.trainers.xgboost_templates import get_template
    default_config = get_template(model_type)

    session.selections['xgboost_config'] = default_config
    session.selections['xgboost_model_type'] = model_type  # Store for later
    await self.state_manager.update_session(session)

    # Start with n_estimators configuration
    keyboard = [
        [InlineKeyboardButton("50 trees", callback_data="xgboost_n_estimators:50")],
        [InlineKeyboardButton("100 trees (recommended)", callback_data="xgboost_n_estimators:100")],
        [InlineKeyboardButton("200 trees", callback_data="xgboost_n_estimators:200")],
        [InlineKeyboardButton("Custom", callback_data="xgboost_n_estimators:custom")]
    ]
    add_back_button(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🧠 **XGBoost Configuration**\n\n"
        "**Step 1/5: Number of Boosting Rounds**\n\n"
        "How many trees to build?\n"
        "(More trees = better fit, but slower training)",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_xgboost_n_estimators(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle XGBoost n_estimators selection."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    n_estimators_value = query.data.split(":")[-1]

    session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

    # Handle custom input
    if n_estimators_value == "custom":
        # TODO: Implement custom value input workflow
        n_estimators = 100  # Default for now
    else:
        n_estimators = int(n_estimators_value)

    session.selections['xgboost_config']['n_estimators'] = n_estimators
    await self.state_manager.update_session(session)

    # Move to max_depth selection
    keyboard = [
        [InlineKeyboardButton("3 levels", callback_data="xgboost_max_depth:3")],
        [InlineKeyboardButton("6 levels (recommended)", callback_data="xgboost_max_depth:6")],
        [InlineKeyboardButton("9 levels", callback_data="xgboost_max_depth:9")],
        [InlineKeyboardButton("Custom", callback_data="xgboost_max_depth:custom")]
    ]
    add_back_button(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🧠 **XGBoost Configuration**\n\n"
        "**Step 2/5: Maximum Tree Depth**\n\n"
        "How deep should each tree be?\n"
        "(Deeper = more complex patterns, risk of overfitting)",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_xgboost_max_depth(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle XGBoost max_depth selection."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    max_depth_value = query.data.split(":")[-1]

    session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

    if max_depth_value == "custom":
        max_depth = 6  # Default for now
    else:
        max_depth = int(max_depth_value)

    session.selections['xgboost_config']['max_depth'] = max_depth
    await self.state_manager.update_session(session)

    # Move to learning_rate selection
    keyboard = [
        [InlineKeyboardButton("0.01 (conservative)", callback_data="xgboost_learning_rate:0.01")],
        [InlineKeyboardButton("0.1 (recommended)", callback_data="xgboost_learning_rate:0.1")],
        [InlineKeyboardButton("0.3 (aggressive)", callback_data="xgboost_learning_rate:0.3")],
        [InlineKeyboardButton("Custom", callback_data="xgboost_learning_rate:custom")]
    ]
    add_back_button(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🧠 **XGBoost Configuration**\n\n"
        "**Step 3/5: Learning Rate**\n\n"
        "How fast should the model learn?\n"
        "(Lower = more stable, but needs more trees)",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_xgboost_learning_rate(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle XGBoost learning_rate selection."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    learning_rate_value = query.data.split(":")[-1]

    session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

    if learning_rate_value == "custom":
        learning_rate = 0.1  # Default for now
    else:
        learning_rate = float(learning_rate_value)

    session.selections['xgboost_config']['learning_rate'] = learning_rate
    await self.state_manager.update_session(session)

    # Move to subsample selection
    keyboard = [
        [InlineKeyboardButton("0.6 (60%)", callback_data="xgboost_subsample:0.6")],
        [InlineKeyboardButton("0.8 (80% - recommended)", callback_data="xgboost_subsample:0.8")],
        [InlineKeyboardButton("1.0 (100% - all data)", callback_data="xgboost_subsample:1.0")],
        [InlineKeyboardButton("Custom", callback_data="xgboost_subsample:custom")]
    ]
    add_back_button(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🧠 **XGBoost Configuration**\n\n"
        "**Step 4/5: Subsample Ratio**\n\n"
        "What fraction of data to use per tree?\n"
        "(Lower = more diverse trees, prevents overfitting)",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_xgboost_subsample(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle XGBoost subsample selection."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    subsample_value = query.data.split(":")[-1]

    session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

    if subsample_value == "custom":
        subsample = 0.8  # Default for now
    else:
        subsample = float(subsample_value)

    session.selections['xgboost_config']['subsample'] = subsample
    await self.state_manager.update_session(session)

    # Move to colsample_bytree selection
    keyboard = [
        [InlineKeyboardButton("0.6 (60%)", callback_data="xgboost_colsample:0.6")],
        [InlineKeyboardButton("0.8 (80% - recommended)", callback_data="xgboost_colsample:0.8")],
        [InlineKeyboardButton("1.0 (100% - all features)", callback_data="xgboost_colsample:1.0")],
        [InlineKeyboardButton("Custom", callback_data="xgboost_colsample:custom")]
    ]
    add_back_button(keyboard)
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        "🧠 **XGBoost Configuration**\n\n"
        "**Step 5/5: Column Subsample Ratio**\n\n"
        "What fraction of features to use per tree?\n"
        "(Lower = more diverse trees, prevents overfitting)",
        reply_markup=reply_markup,
        parse_mode="Markdown"
    )

async def handle_xgboost_colsample(
    self,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle XGBoost colsample_bytree selection and start training."""
    query = update.callback_query
    await query.answer()

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    colsample_value = query.data.split(":")[-1]

    session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

    if colsample_value == "custom":
        colsample = 0.8  # Default for now
    else:
        colsample = float(colsample_value)

    session.selections['xgboost_config']['colsample_bytree'] = colsample
    await self.state_manager.update_session(session)

    # Configuration complete - start training
    model_type = session.selections.get('xgboost_model_type')
    xgboost_config = session.selections.get('xgboost_config')

    await query.edit_message_text(
        f"✅ **XGBoost Configuration Complete**\n\n"
        f"🎯 **Model**: {model_type.replace('_', '\\_')}\n"
        f"⚙️ **Parameters**:\n"
        f"• n\\_estimators: {xgboost_config['n_estimators']}\n"
        f"• max\\_depth: {xgboost_config['max_depth']}\n"
        f"• learning\\_rate: {xgboost_config['learning_rate']}\n"
        f"• subsample: {xgboost_config['subsample']}\n"
        f"• colsample\\_bytree: {xgboost_config['colsample_bytree']}\n\n"
        f"🚀 Starting training...\n\n"
        f"This may take a few moments.",
        parse_mode="Markdown"
    )

    # Execute training with XGBoost config
    await self._execute_sklearn_training(update, context, session, model_type)
```

#### 4. Update _execute_sklearn_training

**File**: `src/bot/ml_handlers/ml_training_local_path.py`
**Location**: Line 1118-1124 (in _execute_sklearn_training)

**Current**:
```python
# Get default hyperparameters based on model type
hyperparameters = None
if model_type.startswith('xgboost_'):
    # Use XGBoost template defaults
    from src.engines.trainers.xgboost_templates import get_template
    hyperparameters = get_template(model_type)
    print(f"🚀 DEBUG: Using XGBoost template hyperparameters: {hyperparameters}")
```

**Updated**:
```python
# Get hyperparameters based on model type
hyperparameters = None
if model_type.startswith('xgboost_'):
    # Check if user configured parameters or use defaults
    xgboost_config = session.selections.get('xgboost_config')
    if xgboost_config:
        # User selected custom parameters
        hyperparameters = xgboost_config
        print(f"🚀 DEBUG: Using user-configured XGBoost parameters: {hyperparameters}")
    else:
        # Use XGBoost template defaults
        from src.engines.trainers.xgboost_templates import get_template
        hyperparameters = get_template(model_type)
        print(f"🚀 DEBUG: Using XGBoost template hyperparameters: {hyperparameters}")
```

#### 5. Register Callback Handlers

**File**: `src/bot/ml_handlers/ml_training_local_path.py`
**Location**: Lines 2356-2400 (register_local_path_handlers function)

**Add After Keras Handlers**:
```python
# XGBoost configuration callback handlers
application.add_handler(
    CallbackQueryHandler(
        handler.handle_xgboost_n_estimators,
        pattern=r"^xgboost_n_estimators:"
    )
)

application.add_handler(
    CallbackQueryHandler(
        handler.handle_xgboost_max_depth,
        pattern=r"^xgboost_max_depth:"
    )
)

application.add_handler(
    CallbackQueryHandler(
        handler.handle_xgboost_learning_rate,
        pattern=r"^xgboost_learning_rate:"
    )
)

application.add_handler(
    CallbackQueryHandler(
        handler.handle_xgboost_subsample,
        pattern=r"^xgboost_subsample:"
    )
)

application.add_handler(
    CallbackQueryHandler(
        handler.handle_xgboost_colsample,
        pattern=r"^xgboost_colsample:"
    )
)
```

## Implementation Steps

### Phase 1: Core Infrastructure (2-3 hours)

1. **Update Model Selection Menu** ✅
   - Rename "Gradient Boosting" → "Gradient Boosting (sklearn)"
   - Verify XGBoost options already present
   - Test: Click each option, verify correct model_type passed

2. **Implement _start_xgboost_config** ✅
   - Create method to initialize config workflow
   - Load defaults from xgboost_templates
   - Display Step 1/5 (n_estimators)
   - Test: Start XGBoost workflow, verify UI shows

3. **Implement 5 Parameter Handlers** ✅
   - `handle_xgboost_n_estimators`
   - `handle_xgboost_max_depth`
   - `handle_xgboost_learning_rate`
   - `handle_xgboost_subsample`
   - `handle_xgboost_colsample`
   - Test: Complete full workflow, verify parameters stored

4. **Update handle_model_selection** ✅
   - Add XGBoost detection logic
   - Route to _start_xgboost_config
   - Test: Click XGBoost → parameter workflow starts

5. **Update _execute_sklearn_training** ✅
   - Check for user-configured parameters
   - Use custom config if present, else defaults
   - Test: Training uses correct parameters

### Phase 2: Callback Registration (30 min)

6. **Register XGBoost Handlers** ✅
   - Add 5 callback handlers in register_local_path_handlers
   - Test: All callbacks work, no conflicts

### Phase 3: Testing & Validation (2 hours)

7. **Unit Tests** ✅
   - Test parameter validation
   - Test default vs custom config
   - Test session state management

8. **Integration Tests** ✅
   - Test complete XGBoost workflow
   - Test sklearn gradient_boosting still works
   - Test XGBoost training with custom params
   - Test binary classification, multiclass, regression

9. **Manual Testing** ✅
   - Test via Telegram UI
   - Verify all 5 parameter steps work
   - Verify training completes
   - Verify metrics display correctly

## Files to Modify

### Primary File
**`src/bot/ml_handlers/ml_training_local_path.py`**
- Line 934: Update button label
- Line 937: Verify XGBoost option present
- Line 1023: Add XGBoost detection
- After Line 1070: Add `_start_xgboost_config` method (~70 lines)
- After Line 1140: Add 5 parameter handlers (~250 lines)
- Line 1120: Update hyperparameter logic
- Line 2400: Register 5 new callbacks

### Supporting Files
**`src/engines/trainers/xgboost_templates.py`**
- Already exists, no changes needed
- Provides default configurations

**`tests/unit/test_xgboost_trainer.py`**
- Already exists with 21 tests
- May add parameter validation tests

**`tests/integration/test_xgboost_workflow.py`**
- Add test for parameter configuration workflow
- Add test for custom vs default parameters

## Testing Plan

### Unit Tests

```python
# tests/unit/test_xgboost_config.py

def test_xgboost_config_initialization():
    """Test XGBoost config initializes with defaults."""
    from src.engines.trainers.xgboost_templates import get_template

    config = get_template('xgboost_binary_classification')

    assert config['n_estimators'] == 100
    assert config['max_depth'] == 6
    assert config['learning_rate'] == 0.1
    assert config['subsample'] == 0.8
    assert config['colsample_bytree'] == 0.8

def test_custom_parameter_override():
    """Test custom parameters override defaults."""
    custom_config = {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.01,
        'subsample': 0.6,
        'colsample_bytree': 1.0
    }

    # Verify custom values preserved
    assert custom_config['n_estimators'] == 200
    assert custom_config['learning_rate'] == 0.01
```

### Integration Tests

```python
# tests/integration/test_xgboost_parameter_workflow.py

@pytest.mark.asyncio
async def test_xgboost_parameter_workflow():
    """Test complete XGBoost parameter configuration workflow."""
    # 1. Select XGBoost model
    # 2. Configure n_estimators
    # 3. Configure max_depth
    # 4. Configure learning_rate
    # 5. Configure subsample
    # 6. Configure colsample_bytree
    # 7. Verify training starts
    # 8. Verify custom parameters used
    pass

@pytest.mark.asyncio
async def test_sklearn_gradient_boosting_unchanged():
    """Test sklearn gradient_boosting still works as before."""
    # 1. Select "Gradient Boosting (sklearn)"
    # 2. Verify immediate training (no parameter config)
    # 3. Verify sklearn model trained
    pass
```

## Acceptance Criteria

### Functional Requirements
- ✅ User can select between sklearn and XGBoost gradient boosting
- ✅ XGBoost models trigger parameter configuration workflow
- ✅ All 5 parameters configurable (n_estimators, max_depth, learning_rate, subsample, colsample_bytree)
- ✅ Training uses user-selected parameters
- ✅ sklearn gradient_boosting still works without parameter config
- ✅ Metrics display correctly after training
- ✅ Model naming workflow triggers after training

### Non-Functional Requirements
- ✅ Parameter workflow similar UX to Keras models
- ✅ Back button works at each step
- ✅ Default values clearly marked as "recommended"
- ✅ Custom value input available (Phase 2 enhancement)
- ✅ Parameter validation prevents invalid values
- ✅ Training time reasonable (<30 seconds for typical datasets)

### Quality Requirements
- ✅ 100% test coverage for new handlers
- ✅ No regression in existing sklearn models
- ✅ No Markdown formatting errors
- ✅ All debug logs present for troubleshooting
- ✅ Code follows existing patterns (Keras handler structure)

## Risk Assessment

### High Risk
- **Backward Compatibility**: sklearn gradient_boosting must continue working
  - **Mitigation**: Thorough testing of sklearn model selection

### Medium Risk
- **Parameter Validation**: Invalid parameters could cause training failures
  - **Mitigation**: Use xgboost_templates defaults as validation reference

- **Session State Management**: Config must persist across workflow steps
  - **Mitigation**: Follow Keras config pattern (session.selections['xgboost_config'])

### Low Risk
- **UI Consistency**: Parameter screens must match Keras workflow style
  - **Mitigation**: Copy Keras handler structure exactly

## Future Enhancements

### Phase 2 (Custom Values)
- Implement custom value input for each parameter
- Add parameter validation (ranges, types)
- Show parameter descriptions/tooltips

### Phase 3 (Advanced Config)
- Add more XGBoost parameters (gamma, min_child_weight, reg_alpha, reg_lambda)
- Parameter presets (conservative, balanced, aggressive)
- Save parameter configurations as templates

### Phase 4 (Auto-tuning)
- Hyperparameter search integration
- Cross-validation during parameter selection
- Parameter importance analysis

## References

- **Existing Code**: `src/bot/ml_handlers/ml_training_local_path.py` (Keras workflow lines 1043-1800)
- **XGBoost Docs**: https://xgboost.readthedocs.io/en/stable/parameter.html
- **Template System**: `src/engines/trainers/xgboost_templates.py`
- **Similar Implementation**: Keras parameter configuration (lines 1043-1800)

## Appendix: Code Examples

### Example Session State After Configuration

```python
session.selections = {
    'target_column': 'class',
    'feature_columns': ['feature1', 'feature2', ..., 'feature20'],
    'task_type': 'classification',
    'model_type': 'xgboost_binary_classification',
    'xgboost_model_type': 'xgboost_binary_classification',  # Store for training
    'xgboost_config': {
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        # Additional template defaults
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    }
}
```

### Example Training Call

```python
result = await self.ml_engine.train_model(
    file_path=file_path,
    task_type='classification',
    model_type='xgboost_binary_classification',
    target_column='class',
    feature_columns=['feature1', 'feature2', ...],
    user_id=user_id,
    hyperparameters={
        'n_estimators': 200,
        'max_depth': 9,
        'learning_rate': 0.01,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42
    },
    test_size=0.2
)
```

---

**Implementation Start Date**: TBD
**Estimated Completion**: TBD
**Implemented By**: TBD
