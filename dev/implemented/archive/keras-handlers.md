# Keras-Telegram Integration Plan

**Date**: 2025-10-04
**Status**: 📋 **PLANNING COMPLETE - READY FOR IMPLEMENTATION**

---

## Executive Summary

**Problem**: Keras neural network models are fully implemented in ML Engine but unreachable from the Telegram bot interface. Users cannot train Keras models through conversations.

**Root Cause**: `src/bot/workflow_handlers.py` only recognizes sklearn model types (linear, random_forest, neural_network). The model_type_map (lines 338-353) has no Keras entries.

**Solution**: Extend Telegram workflow with conditional state branching. sklearn models maintain simple 1-step flow, Keras models add 2 states for architecture specification and hyperparameter collection.

**Impact**: ~500 lines of new code across 5 files. Zero impact on existing sklearn workflow.

---

## Gap Analysis

### Current State

**ML Engine Support** (✅ Complete):
- ✅ `KerasNeuralNetworkTrainer` implemented (410 lines)
- ✅ `ModelManager` supports Keras save/load (JSON + H5)
- ✅ `MLEngine.train_model()` routes Keras models correctly
- ✅ Test coverage: 4/4 Keras workflow tests passing

**Telegram Bot Support** (❌ Missing):
- ❌ No Keras model types in `model_type_map`
- ❌ No architecture specification state
- ❌ No hyperparameter collection state
- ❌ No Keras-specific user prompts

### Evidence

**File**: `src/bot/workflow_handlers.py`
**Lines**: 338-353
**Current model_type_map**:
```python
model_type_map = {
    'linear': 'linear',
    'random': 'random_forest',
    'neural': 'neural_network',  # sklearn MLP only
    'auto': 'auto'
}
```

**Missing**: `keras_binary_classification`, `keras_multiclass_classification`, `keras_regression`

---

## Integration Strategy

### Conditional State Branching

**Principle**: Preserve simple sklearn flow, add complexity only for Keras models.

**Implementation**: Use `is_keras_model(model_type)` helper to branch state transitions.

**State Flow Comparison**:

```
sklearn models:
CONFIRMING_MODEL → TRAINING (1 step)

Keras models:
CONFIRMING_MODEL → SPECIFYING_ARCHITECTURE → COLLECTING_HYPERPARAMETERS → TRAINING (3 steps)
```

### Architecture Specification Challenge

Keras requires JSON architecture that sklearn doesn't need:
```python
architecture = {
    "layers": [
        {"type": "Dense", "units": 14, "activation": "relu", "kernel_initializer": "random_normal"},
        {"type": "Dense", "units": 1, "activation": "sigmoid"}
    ],
    "compile": {"loss": "binary_crossentropy", "optimizer": "adam", "metrics": ["accuracy"]}
}
```

**Solution**: Provide templates for beginners, allow JSON upload for advanced users.

### Hyperparameter Requirements

Keras training requires parameters that sklearn auto-selects:
- `epochs`: 10-1000+ (user must specify, no good default)
- `batch_size`: 16-128 typical (default: 32)
- `verbose`: 0/1 (default: 1 for Telegram)
- `validation_split`: 0.0-0.3 (optional, default: 0.0)

**Solution**: Multi-turn conversation to collect epochs and batch_size.

---

## User Experience Flow

### Complete Conversation Example

```
1. User: "/train"
   Bot: "Please upload your training data (CSV format)"
   State: AWAITING_DATA

2. User: [uploads housing.csv]
   Bot: "Data uploaded: 1000 rows, 15 columns
        Select target column:
        1) sqft
        2) bedrooms
        3) bathrooms
        ...
        14) age
        15) price"
   State: SELECTING_TARGET

3. User: "15"
   Bot: "Target: price
        Select features (comma-separated numbers):
        1) sqft
        2) bedrooms
        ...
        14) age"
   State: SELECTING_FEATURES

4. User: "1,2,3,4,5,6,7,8,9,10,11,12,13,14"
   Bot: "Features selected: sqft, bedrooms, bathrooms, ... (14 total)

        Choose model type:
        • linear - Linear Regression
        • random - Random Forest
        • keras_binary - Keras Binary Classification (NN)
        • keras_multi - Keras Multiclass Classification (NN)
        • keras_reg - Keras Regression (NN)
        • auto - Automatic selection

        Enter choice:"
   State: CONFIRMING_MODEL

5. User: "keras_binary"
   Bot: "Keras Binary Classification selected

        Choose architecture:
        1) Default template (14→14→1, sigmoid output)
        2) Custom JSON (advanced)

        Enter choice:"
   State: SPECIFYING_ARCHITECTURE [NEW]

6. User: "1"
   Bot: "Default architecture selected:
        • Input: 14 features
        • Hidden: Dense(14, relu, random_normal)
        • Output: Dense(1, sigmoid)
        • Loss: binary_crossentropy
        • Optimizer: adam

        How many epochs? (Recommended: 100-500 for this dataset size)"
   State: COLLECTING_HYPERPARAMETERS [NEW]

7. User: "300"
   Bot: "Epochs: 300
        Batch size? (Recommended: 32-128, default: 32)"

8. User: "70"
   Bot: "Training configuration:
        • Model: Keras Binary Classification
        • Epochs: 300
        • Batch size: 70
        • Architecture: Default template

        Starting training... ⏳"
   State: TRAINING

9. [Training executes asynchronously]

10. Bot: "✅ Model trained successfully!

         Model ID: model_12345_keras_binary_classification_20251004_143022

         Performance Metrics:
         • Accuracy: 92.34%
         • Loss: 0.1567
         • Precision: 91.2%
         • Recall: 93.5%
         • F1-Score: 92.3%

         Confusion Matrix:
         Predicted →  0    1
         Actual ↓
         0           45    5
         1            3   47

         Training time: 12.3s

         Use /predict with this model ID to make predictions on new data."
    State: COMPLETE
```

---

## Implementation Plan

### Phase 1: State Manager Updates

**File**: `src/core/state_manager.py`
**Lines to modify**: ~30
**Changes**:

Add two new states to `MLTrainingState` enum:

```python
class MLTrainingState(Enum):
    AWAITING_DATA = "awaiting_data"
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIRMING_MODEL = "confirming_model"
    SPECIFYING_ARCHITECTURE = "specifying_architecture"  # NEW - Keras only
    COLLECTING_HYPERPARAMETERS = "collecting_hyperparameters"  # NEW - Keras only
    TRAINING = "training"
    COMPLETE = "complete"
```

**Rationale**: Separate states for architecture and hyperparameters provide clear user experience and error recovery points.

---

### Phase 2: Architecture Template System

**File**: `src/engines/trainers/keras_templates.py` (NEW FILE)
**Lines**: ~100
**Purpose**: Provide beginner-friendly architecture templates

**Implementation**:

```python
"""
Keras Architecture Templates.

Provides pre-configured neural network architectures for common tasks.
"""

from typing import Dict, Any, Literal


def get_binary_classification_template(
    n_features: int,
    kernel_initializer: str = "random_normal"
) -> Dict[str, Any]:
    """
    Get template for binary classification.

    Architecture:
    - Input: n_features
    - Hidden: Dense(n_features, relu)
    - Output: Dense(1, sigmoid)
    - Loss: binary_crossentropy

    Args:
        n_features: Number of input features
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict ready for KerasNeuralNetworkTrainer
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "units": n_features,
                "activation": "relu",
                "kernel_initializer": kernel_initializer
            },
            {
                "type": "Dense",
                "units": 1,
                "activation": "sigmoid",
                "kernel_initializer": kernel_initializer
            }
        ],
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"]
        }
    }


def get_multiclass_classification_template(
    n_features: int,
    n_classes: int,
    kernel_initializer: str = "glorot_uniform"
) -> Dict[str, Any]:
    """
    Get template for multiclass classification.

    Architecture:
    - Input: n_features
    - Hidden: Dense(n_features, relu)
    - Output: Dense(n_classes, softmax)
    - Loss: categorical_crossentropy

    Args:
        n_features: Number of input features
        n_classes: Number of output classes
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict ready for KerasNeuralNetworkTrainer
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "units": n_features,
                "activation": "relu",
                "kernel_initializer": kernel_initializer
            },
            {
                "type": "Dense",
                "units": n_classes,
                "activation": "softmax",
                "kernel_initializer": kernel_initializer
            }
        ],
        "compile": {
            "loss": "categorical_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"]
        }
    }


def get_regression_template(
    n_features: int,
    kernel_initializer: str = "glorot_uniform"
) -> Dict[str, Any]:
    """
    Get template for regression.

    Architecture:
    - Input: n_features
    - Hidden: Dense(n_features, relu)
    - Output: Dense(1, linear)
    - Loss: mse

    Args:
        n_features: Number of input features
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict ready for KerasNeuralNetworkTrainer
    """
    return {
        "layers": [
            {
                "type": "Dense",
                "units": n_features,
                "activation": "relu",
                "kernel_initializer": kernel_initializer
            },
            {
                "type": "Dense",
                "units": 1,
                "activation": "linear",
                "kernel_initializer": kernel_initializer
            }
        ],
        "compile": {
            "loss": "mse",
            "optimizer": "adam",
            "metrics": ["mae"]
        }
    }


def get_template(
    model_type: Literal["keras_binary_classification", "keras_multiclass_classification", "keras_regression"],
    n_features: int,
    n_classes: int = 2,
    kernel_initializer: str = "random_normal"
) -> Dict[str, Any]:
    """
    Get architecture template based on model type.

    Args:
        model_type: Type of Keras model
        n_features: Number of input features
        n_classes: Number of classes (for classification)
        kernel_initializer: Weight initialization method

    Returns:
        Architecture dict

    Raises:
        ValueError: If model_type is invalid
    """
    if model_type == "keras_binary_classification":
        return get_binary_classification_template(n_features, kernel_initializer)
    elif model_type == "keras_multiclass_classification":
        return get_multiclass_classification_template(n_features, n_classes, kernel_initializer)
    elif model_type == "keras_regression":
        return get_regression_template(n_features, kernel_initializer)
    else:
        raise ValueError(f"Unknown Keras model type: {model_type}")
```

**Features**:
- Auto-detects n_features from selected feature columns
- Auto-detects n_classes from target column for multiclass
- Provides sensible defaults for activation functions and loss
- Supports kernel_initializer customization (advanced users)

---

### Phase 3: Workflow Handler Extensions

**File**: `src/bot/workflow_handlers.py`
**Lines to add**: ~150
**Changes**:

#### 3.1 Update model_type_map

```python
# Line ~340 (current location)
model_type_map = {
    # sklearn models (existing)
    'linear': 'linear',
    'random': 'random_forest',
    'neural': 'neural_network',  # Keep for sklearn MLP backward compatibility
    'auto': 'auto',

    # Keras models (NEW)
    'keras_binary': 'keras_binary_classification',
    'keras_multi': 'keras_multiclass_classification',
    'keras_reg': 'keras_regression'
}
```

#### 3.2 Add helper function

```python
def is_keras_model(model_type: str) -> bool:
    """
    Check if model type is a Keras model.

    Args:
        model_type: Model type string

    Returns:
        True if Keras model, False otherwise
    """
    keras_models = [
        'keras_binary_classification',
        'keras_multiclass_classification',
        'keras_regression'
    ]
    return model_type in keras_models
```

#### 3.3 Add architecture specification handler

```python
async def handle_architecture_specification(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: UserSession
) -> None:
    """
    Handle architecture specification for Keras models.

    States:
    - Input: User choice (1=template, 2=custom JSON)
    - Output: Store architecture in session.selections
    - Next: COLLECTING_HYPERPARAMETERS
    """
    user_input = update.message.text.strip()

    try:
        choice = int(user_input)

        if choice == 1:
            # Default template
            model_type = session.selections['model_type']
            n_features = len(session.selections['feature_columns'])

            # Auto-detect n_classes for multiclass
            n_classes = 2
            if model_type == 'keras_multiclass_classification':
                # Get unique values in target column from session data
                target_col = session.selections['target_column']
                data = session.data  # Assuming data is stored in session
                n_classes = data[target_col].nunique()

            # Import template function
            from src.engines.trainers.keras_templates import get_template

            architecture = get_template(
                model_type=model_type,
                n_features=n_features,
                n_classes=n_classes
            )

            session.selections['architecture'] = architecture

            # Show architecture summary
            await update.message.reply_text(
                f"Default architecture selected:\n"
                f"• Input: {n_features} features\n"
                f"• Hidden: Dense({n_features}, relu)\n"
                f"• Output: Dense({architecture['layers'][-1]['units']}, "
                f"{architecture['layers'][-1]['activation']})\n"
                f"• Loss: {architecture['compile']['loss']}\n"
                f"• Optimizer: {architecture['compile']['optimizer']}\n\n"
                f"How many epochs? (Recommended: 100-500 for this dataset size)"
            )

            # Transition to hyperparameter collection
            session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS
            session.selections['hyperparam_step'] = 'epochs'  # Track sub-step

        elif choice == 2:
            # Custom JSON architecture
            await update.message.reply_text(
                "Please send your architecture as JSON.\n\n"
                "Example format:\n"
                "```json\n"
                "{\n"
                '  "layers": [\n'
                '    {"type": "Dense", "units": 14, "activation": "relu"},\n'
                '    {"type": "Dense", "units": 1, "activation": "sigmoid"}\n'
                '  ],\n'
                '  "compile": {\n'
                '    "loss": "binary_crossentropy",\n'
                '    "optimizer": "adam",\n'
                '    "metrics": ["accuracy"]\n'
                '  }\n'
                "}\n"
                "```"
            )
            session.selections['expecting_json'] = True

        else:
            await update.message.reply_text(
                "Invalid choice. Please enter 1 for default template or 2 for custom JSON."
            )

    except ValueError:
        # Check if this is JSON input (for custom architecture)
        if session.selections.get('expecting_json', False):
            try:
                import json
                architecture = json.loads(user_input)

                # Validate architecture structure
                if 'layers' not in architecture or 'compile' not in architecture:
                    raise ValueError("Architecture must contain 'layers' and 'compile' keys")

                session.selections['architecture'] = architecture
                session.selections['expecting_json'] = False

                await update.message.reply_text(
                    f"Custom architecture accepted: {len(architecture['layers'])} layers\n\n"
                    f"How many epochs? (Recommended: 100-500)"
                )

                session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS
                session.selections['hyperparam_step'] = 'epochs'

            except json.JSONDecodeError as e:
                await update.message.reply_text(
                    f"Invalid JSON format: {e}\n\n"
                    f"Please send valid JSON or type 1 to use default template."
                )
        else:
            await update.message.reply_text(
                "Please enter a number (1 or 2)."
            )
```

#### 3.4 Add hyperparameter collection handler

```python
async def handle_hyperparameter_collection(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: UserSession
) -> None:
    """
    Handle hyperparameter collection for Keras models.

    Multi-turn conversation:
    1. epochs (required)
    2. batch_size (optional, default 32)

    States:
    - Input: User-provided hyperparameter values
    - Output: Store in session.selections['hyperparameters']
    - Next: TRAINING
    """
    user_input = update.message.text.strip()
    current_step = session.selections.get('hyperparam_step', 'epochs')

    if 'hyperparameters' not in session.selections:
        session.selections['hyperparameters'] = {
            'verbose': 1,  # Always 1 for Telegram
            'validation_split': 0.0  # Default: no validation split
        }

    try:
        if current_step == 'epochs':
            epochs = int(user_input)

            # Validate epochs
            if epochs < 1 or epochs > 10000:
                await update.message.reply_text(
                    "Epochs must be between 1 and 10000.\n"
                    "Recommended: 100-500 for most datasets.\n"
                    "Please enter epochs:"
                )
                return

            session.selections['hyperparameters']['epochs'] = epochs

            # Move to batch_size
            await update.message.reply_text(
                f"Epochs: {epochs}\n"
                f"Batch size? (Recommended: 32-128, default: 32)"
            )
            session.selections['hyperparam_step'] = 'batch_size'

        elif current_step == 'batch_size':
            batch_size = int(user_input)

            # Validate batch_size
            if batch_size < 1:
                await update.message.reply_text(
                    "Batch size must be at least 1.\n"
                    "Recommended: 32-128.\n"
                    "Please enter batch size:"
                )
                return

            session.selections['hyperparameters']['batch_size'] = batch_size

            # Show summary and start training
            architecture = session.selections['architecture']
            hyperparams = session.selections['hyperparameters']

            await update.message.reply_text(
                f"Training configuration:\n"
                f"• Model: {session.selections['model_type']}\n"
                f"• Epochs: {hyperparams['epochs']}\n"
                f"• Batch size: {hyperparams['batch_size']}\n"
                f"• Architecture: {len(architecture['layers'])} layers\n\n"
                f"Starting training... ⏳"
            )

            # Transition to training
            session.current_state = MLTrainingState.TRAINING

            # Trigger actual training (delegate to existing training logic)
            await execute_ml_training(update, context, session)

    except ValueError:
        await update.message.reply_text(
            f"Please enter a valid number for {current_step}."
        )
```

#### 3.5 Update state router

Modify existing state routing logic to include new states:

```python
# In main workflow router function (around line 100-150)
async def route_ml_training_workflow(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: UserSession
) -> None:
    """Route to appropriate handler based on current state."""

    state = session.current_state

    if state == MLTrainingState.AWAITING_DATA:
        await handle_data_upload(update, context, session)
    elif state == MLTrainingState.SELECTING_TARGET:
        await handle_target_selection(update, context, session)
    elif state == MLTrainingState.SELECTING_FEATURES:
        await handle_feature_selection(update, context, session)
    elif state == MLTrainingState.CONFIRMING_MODEL:
        await handle_model_confirmation(update, context, session)
    elif state == MLTrainingState.SPECIFYING_ARCHITECTURE:  # NEW
        await handle_architecture_specification(update, context, session)
    elif state == MLTrainingState.COLLECTING_HYPERPARAMETERS:  # NEW
        await handle_hyperparameter_collection(update, context, session)
    elif state == MLTrainingState.TRAINING:
        # Training in progress, ignore user input
        pass
    elif state == MLTrainingState.COMPLETE:
        # Workflow complete
        pass
```

#### 3.6 Update model confirmation handler

Modify existing `handle_model_confirmation()` to branch based on model type:

```python
async def handle_model_confirmation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: UserSession
) -> None:
    """Handle model type confirmation and route to appropriate next state."""
    user_input = update.message.text.strip().lower()

    # Map user input to model_type
    if user_input in model_type_map:
        model_type = model_type_map[user_input]
        session.selections['model_type'] = model_type

        # Branch based on model type
        if is_keras_model(model_type):
            # Keras models: go to architecture specification
            await update.message.reply_text(
                f"Keras model selected: {model_type}\n\n"
                f"Choose architecture:\n"
                f"1) Default template (recommended for beginners)\n"
                f"2) Custom JSON (advanced)\n\n"
                f"Enter choice:"
            )
            session.current_state = MLTrainingState.SPECIFYING_ARCHITECTURE
        else:
            # sklearn models: go directly to training
            await update.message.reply_text(
                f"Model selected: {model_type}\n"
                f"Starting training... ⏳"
            )
            session.current_state = MLTrainingState.TRAINING
            await execute_ml_training(update, context, session)
    else:
        await update.message.reply_text(
            f"Unknown model type: {user_input}\n"
            f"Please choose from: {', '.join(model_type_map.keys())}"
        )
```

---

### Phase 4: Integration Testing

**File**: `tests/integration/test_keras_telegram_workflow.py` (NEW FILE)
**Lines**: ~200
**Purpose**: End-to-end Telegram conversation flow validation

**Test Coverage**:

```python
"""
Integration tests for Keras-Telegram workflow.

Tests complete user conversations from /train to model completion.
"""

import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from telegram import Update, Message, User, Chat

from src.bot.workflow_handlers import (
    route_ml_training_workflow,
    MLTrainingState
)
from src.core.state_manager import StateManager, UserSession


@pytest.fixture
def mock_update():
    """Create mock Telegram update."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.message = MagicMock(spec=Message)
    update.message.reply_text = AsyncMock()
    return update


@pytest.fixture
def sample_data():
    """Create sample binary classification dataset."""
    return pd.DataFrame({
        'feature1': [1, 2, 3, 4, 5] * 20,
        'feature2': [10, 20, 30, 40, 50] * 20,
        'target': [0, 1, 0, 1, 0] * 20
    })


@pytest.mark.asyncio
class TestKerasTelegramWorkflow:
    """Test complete Keras training workflow through Telegram."""

    async def test_keras_binary_default_template_flow(
        self,
        mock_update,
        sample_data
    ):
        """Test Keras binary classification with default template."""
        state_manager = StateManager()
        session = state_manager.create_session(
            user_id=12345,
            workflow_type='ml_training'
        )

        # Store data in session
        session.data = sample_data

        # Step 1: Select target
        session.current_state = MLTrainingState.SELECTING_TARGET
        mock_update.message.text = "3"  # Select 'target' column
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['target_column'] == 'target'

        # Step 2: Select features
        session.current_state = MLTrainingState.SELECTING_FEATURES
        mock_update.message.text = "1,2"  # Select feature1, feature2
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['feature_columns'] == ['feature1', 'feature2']

        # Step 3: Confirm Keras model
        session.current_state = MLTrainingState.CONFIRMING_MODEL
        mock_update.message.text = "keras_binary"
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['model_type'] == 'keras_binary_classification'
        assert session.current_state == MLTrainingState.SPECIFYING_ARCHITECTURE

        # Step 4: Choose default template
        mock_update.message.text = "1"
        await route_ml_training_workflow(mock_update, None, session)
        assert 'architecture' in session.selections
        assert session.current_state == MLTrainingState.COLLECTING_HYPERPARAMETERS

        # Verify architecture structure
        arch = session.selections['architecture']
        assert 'layers' in arch
        assert 'compile' in arch
        assert len(arch['layers']) == 2  # Input hidden + output
        assert arch['layers'][-1]['activation'] == 'sigmoid'  # Binary classification

        # Step 5: Provide epochs
        mock_update.message.text = "100"
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['hyperparameters']['epochs'] == 100
        assert session.selections['hyperparam_step'] == 'batch_size'

        # Step 6: Provide batch_size
        mock_update.message.text = "32"
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['hyperparameters']['batch_size'] == 32
        assert session.current_state == MLTrainingState.TRAINING

    async def test_keras_custom_json_architecture(
        self,
        mock_update,
        sample_data
    ):
        """Test Keras with custom JSON architecture."""
        state_manager = StateManager()
        session = state_manager.create_session(
            user_id=12345,
            workflow_type='ml_training'
        )
        session.data = sample_data
        session.selections = {
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2'],
            'model_type': 'keras_binary_classification'
        }

        # Step 1: Choose custom JSON
        session.current_state = MLTrainingState.SPECIFYING_ARCHITECTURE
        mock_update.message.text = "2"
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['expecting_json'] == True

        # Step 2: Provide JSON architecture
        custom_arch = '''{
            "layers": [
                {"type": "Dense", "units": 10, "activation": "relu"},
                {"type": "Dropout", "rate": 0.5},
                {"type": "Dense", "units": 1, "activation": "sigmoid"}
            ],
            "compile": {
                "loss": "binary_crossentropy",
                "optimizer": "adam",
                "metrics": ["accuracy"]
            }
        }'''
        mock_update.message.text = custom_arch
        await route_ml_training_workflow(mock_update, None, session)

        assert 'architecture' in session.selections
        assert len(session.selections['architecture']['layers']) == 3
        assert session.current_state == MLTrainingState.COLLECTING_HYPERPARAMETERS

    async def test_hyperparameter_validation(
        self,
        mock_update,
        sample_data
    ):
        """Test hyperparameter validation (epochs, batch_size bounds)."""
        state_manager = StateManager()
        session = state_manager.create_session(
            user_id=12345,
            workflow_type='ml_training'
        )
        session.current_state = MLTrainingState.COLLECTING_HYPERPARAMETERS
        session.selections = {
            'architecture': {'layers': [], 'compile': {}},
            'hyperparam_step': 'epochs'
        }

        # Test invalid epochs (too low)
        mock_update.message.text = "0"
        await route_ml_training_workflow(mock_update, None, session)
        assert 'epochs' not in session.selections.get('hyperparameters', {})

        # Test invalid epochs (too high)
        mock_update.message.text = "20000"
        await route_ml_training_workflow(mock_update, None, session)
        assert 'epochs' not in session.selections.get('hyperparameters', {})

        # Test valid epochs
        mock_update.message.text = "300"
        await route_ml_training_workflow(mock_update, None, session)
        assert session.selections['hyperparameters']['epochs'] == 300

    async def test_sklearn_backward_compatibility(
        self,
        mock_update,
        sample_data
    ):
        """Test that sklearn models skip Keras-specific states."""
        state_manager = StateManager()
        session = state_manager.create_session(
            user_id=12345,
            workflow_type='ml_training'
        )
        session.data = sample_data
        session.selections = {
            'target_column': 'target',
            'feature_columns': ['feature1', 'feature2']
        }

        # Select sklearn model
        session.current_state = MLTrainingState.CONFIRMING_MODEL
        mock_update.message.text = "random"
        await route_ml_training_workflow(mock_update, None, session)

        # Should skip directly to TRAINING, not SPECIFYING_ARCHITECTURE
        assert session.selections['model_type'] == 'random_forest'
        assert session.current_state == MLTrainingState.TRAINING
        assert 'architecture' not in session.selections
```

---

## Files Modified/Created Summary

### Modified Files (3)

1. **src/core/state_manager.py**
   - Add `SPECIFYING_ARCHITECTURE` state
   - Add `COLLECTING_HYPERPARAMETERS` state
   - Lines added: ~20

2. **src/bot/workflow_handlers.py**
   - Update `model_type_map` with Keras entries
   - Add `is_keras_model()` helper
   - Add `handle_architecture_specification()` handler
   - Add `handle_hyperparameter_collection()` handler
   - Update state router
   - Update `handle_model_confirmation()` for branching
   - Lines added: ~150

3. **src/bot/handlers.py** (minor)
   - Import keras_templates if needed
   - Lines added: ~5

### Created Files (2)

1. **src/engines/trainers/keras_templates.py**
   - Template generation functions
   - Binary classification template
   - Multiclass classification template
   - Regression template
   - Lines: ~100

2. **tests/integration/test_keras_telegram_workflow.py**
   - End-to-end workflow tests
   - Template selection tests
   - Custom JSON tests
   - Validation tests
   - Backward compatibility tests
   - Lines: ~200

**Total Impact**: ~475 lines of new code across 5 files

---

## Backward Compatibility Verification

### ✅ No Breaking Changes

**sklearn workflow preserved**:
- State flow unchanged: `CONFIRMING_MODEL → TRAINING`
- Model type map keys unchanged: `linear`, `random`, `neural`, `auto`
- Session structure unchanged
- No new required parameters

**Branching logic**:
- `is_keras_model()` check determines path
- sklearn models skip `SPECIFYING_ARCHITECTURE` and `COLLECTING_HYPERPARAMETERS`
- Default behavior (unrecognized model): falls back to sklearn path

**Orchestrator**:
- No changes needed - already routes Keras models via `model_type` parameter
- TaskDefinition construction updated only for Keras models

**Result processor**:
- No changes needed - already handles classification metrics from sklearn classifiers
- Keras metrics (accuracy, precision, recall, f1, confusion_matrix) already supported

---

## Error Handling & Validation

### Architecture Validation

**Template validation**:
- Check output layer matches task type:
  - Binary: Dense(1, sigmoid)
  - Multiclass: Dense(n_classes, softmax)
  - Regression: Dense(1, linear)

**JSON validation**:
- Validate JSON syntax
- Validate required keys: `layers`, `compile`
- Validate layer structure
- Provide example on error

**Example error message**:
```
❌ Invalid architecture: Binary classification requires Dense(1, sigmoid) output layer.
Your architecture has Dense(10, relu).

Please fix or use default template (type 1).
```

### Hyperparameter Validation

**Epochs**:
- Range: 1-10000
- Recommended: 100-500 for typical datasets
- Error: "Epochs must be between 1 and 10000"

**Batch size**:
- Range: 1-len(data)
- Recommended: 32-128
- Error: "Batch size must be between 1 and {len(data)}"

**Validation split** (future):
- Range: 0.0-0.5
- Default: 0.0 (no validation split for simplicity)

### Training Failures

**Already handled by MLEngine**:
- OOM errors
- Convergence failures
- Invalid data shapes
- Missing dependencies

**Error propagation**:
- MLEngine exceptions bubble up to workflow handler
- Workflow handler sends friendly message to user
- Session state reset to allow retry

**Example**:
```
❌ Training failed: Out of memory.

Try reducing:
• Batch size (current: 128)
• Number of epochs (current: 1000)
• Model complexity (fewer units/layers)

Type /train to start over.
```

---

## Testing Strategy

### Unit Tests (Existing)

Already covered:
- ✅ KerasNeuralNetworkTrainer (4/4 tests passing)
- ✅ ModelManager Keras save/load
- ✅ MLEngine routing

### Integration Tests (New)

**File**: `tests/integration/test_keras_telegram_workflow.py`

**Coverage**:
1. ✅ Complete workflow with default template
2. ✅ Complete workflow with custom JSON
3. ✅ Hyperparameter validation (epochs, batch_size)
4. ✅ Architecture validation (output layer checks)
5. ✅ sklearn backward compatibility
6. ✅ Error recovery (invalid JSON, out-of-range values)

### Manual Testing Checklist

- [ ] Binary classification with default template
- [ ] Multiclass classification (3+ classes)
- [ ] Regression with Keras
- [ ] Custom JSON architecture with Dropout layers
- [ ] Invalid JSON handling
- [ ] Epochs out of range (0, 20000)
- [ ] Batch size validation
- [ ] Training completion and metrics display
- [ ] Model save/load after Telegram training
- [ ] sklearn models still work (random forest, linear)

---

## Risk Assessment

### Low Risk

✅ **Isolated changes**: All new code in dedicated handlers
✅ **Explicit branching**: `is_keras_model()` check prevents accidental sklearn impact
✅ **Backward compatible**: sklearn workflow untouched
✅ **No orchestrator changes**: Already Keras-ready

### Medium Risk

⚠️ **State machine complexity**: 2 new states increase branching logic
**Mitigation**: Comprehensive unit tests for state transitions

⚠️ **User experience**: Multi-turn conversation may confuse users
**Mitigation**: Clear prompts, examples, defaults

### High Risk

None identified.

---

## Implementation Timeline

### Phase 1: Foundation (Day 1)
- ✅ Create `keras_templates.py` with 3 templates
- ✅ Update `state_manager.py` with 2 new states
- ✅ Write unit tests for templates

### Phase 2: Workflow Handlers (Day 2-3)
- ✅ Update `model_type_map`
- ✅ Implement `is_keras_model()` helper
- ✅ Implement `handle_architecture_specification()`
- ✅ Implement `handle_hyperparameter_collection()`
- ✅ Update state router
- ✅ Update `handle_model_confirmation()`

### Phase 3: Testing (Day 4)
- ✅ Write integration tests
- ✅ Manual testing checklist
- ✅ sklearn backward compatibility verification

### Phase 4: Documentation (Day 5)
- ✅ Update CLAUDE.md with Keras workflow examples
- ✅ Update README with Keras Telegram usage
- ✅ Add inline code comments

**Total Estimate**: 5 days

---

## Success Criteria

### Functional Requirements

✅ Users can train Keras models through Telegram `/train` command
✅ Default templates work for binary/multiclass/regression
✅ Custom JSON architectures accepted and validated
✅ Epochs and batch_size collected and validated
✅ Training completes and displays metrics
✅ Models are saved and loadable for predictions
✅ sklearn workflow remains unchanged (backward compatible)

### Quality Requirements

✅ All unit tests passing (existing + new)
✅ All integration tests passing
✅ No regressions in sklearn model training
✅ Error messages are user-friendly
✅ Code follows project style guidelines
✅ Type annotations complete

### User Experience Requirements

✅ Clear prompts at each step
✅ Examples provided for JSON input
✅ Validation errors are actionable
✅ Training progress visible
✅ Metrics formatted for readability

---

## Future Enhancements (Out of Scope)

These features are NOT included in this integration but could be added later:

1. **Advanced Layer Types**: Conv2D, LSTM, BatchNormalization
2. **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
3. **Learning Curves**: Epoch-by-epoch loss/accuracy visualization
4. **Custom Loss Functions**: User-defined loss
5. **GPU Support**: Automatic GPU detection and usage
6. **Model Comparison**: Train multiple Keras variants and compare
7. **Architecture Search**: AutoML-style architecture optimization
8. **Transfer Learning**: Pre-trained model fine-tuning

---

## Conclusion

**Implementation Ready**: ✅
**Risk Level**: Low
**Estimated Effort**: ~475 lines across 5 files
**Timeline**: 5 days
**Breaking Changes**: None

This plan provides a complete roadmap for integrating Keras neural network training into the Telegram bot workflow while maintaining 100% backward compatibility with existing sklearn models.

**Next Steps**:
1. Review and approve plan
2. Create feature branch: `feature/keras-telegram-integration`
3. Implement Phase 1 (templates + states)
4. Test Phase 1
5. Continue through Phase 4
6. Submit PR with comprehensive tests

---

**Document Version**: 1.0
**Last Updated**: 2025-10-04
**Author**: Statistical Modeling Agent Development Team
