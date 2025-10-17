# Keras Neural Network Integration Plan

**Created**: 2025-10-04
**Status**: Planning Phase
**Priority**: High
**Target**: Enable Keras Sequential model support matching user's existing workflow

---

## Executive Summary

This plan outlines the integration of Keras/TensorFlow neural network capabilities into the ML Engine to support the user's existing Keras training workflow. The current system only supports scikit-learn MLPClassifier/MLPRegressor, but the user requires Keras Sequential API for custom architectures, fine-grained training control, and Keras-native model serialization.

**User's Script**: `test_data/[ML] Fattor Response Model 1 - NN 14var CSV Training v1.py`

---

## 1. User Workflow Analysis

### 1.1 Current User Script Architecture

```python
# User's Keras Sequential Model
model = Sequential()
model.add(Dense(14, input_dim=14, kernel_initializer='random_normal', activation='relu'))
model.add(Dense(1, kernel_initializer='random_normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=300, batch_size=70)

# Save as JSON + H5
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
```

### 1.2 User's Training Configuration

**Multi-Model Training**: 8 variants with different hyperparameters

| Model | Initializer | Epochs | Batch Size |
|-------|------------|--------|------------|
| RES_P1 | random_normal | 300 | 70 |
| RES_P2 | random_normal | 400 | 90 |
| RES_P3 | random_uniform | 300 | 70 |
| RES_P4 | random_uniform | 400 | 90 |
| RES_P5 | normal | 300 | 70 |
| RES_P6 | normal | 400 | 90 |
| RES_P7 | uniform | 300 | 70 |
| RES_P8 | uniform | 400 | 90 |

**Data Preprocessing**:
- StandardScaler (fit_transform on training data)
- No train/test split (trains on 100% of data)
- Position-based column indexing (columns 2-16 for features, 16 for target)

**Model Task**:
- Binary classification (propensity to make promise)
- 14 input features
- 1 output (sigmoid activation)

---

## 2. Gap Analysis

### 2.1 Critical Gaps (Must Fix)

#### GAP-1: TensorFlow/Keras Dependency Missing
**Current State**: `requirements.txt` has no TensorFlow or Keras
**Impact**: Cannot import Keras modules
**Solution**: Add `tensorflow>=2.12.0` to requirements.txt
**Priority**: P0 - Blocker

#### GAP-2: Keras Trainer Class Missing
**Current State**: Only `neural_network_trainer.py` (sklearn-based)
**Impact**: No way to create/train Keras models
**Solution**: Create `src/engines/trainers/keras_trainer.py`
**Priority**: P0 - Blocker

#### GAP-3: Model Serialization Format Incompatible
**Current State**: ModelManager saves as `.pkl` (joblib)
**User Needs**: JSON (architecture) + H5 (weights)
**Impact**: Cannot save/load Keras models
**Solution**: Add Keras-specific save/load logic to ModelManager
**Priority**: P0 - Blocker

#### GAP-4: ML Engine Routing
**Current State**: Only routes to sklearn trainers
**User Needs**: Route to Keras trainer for Keras models
**Impact**: Cannot use Keras models through ML Engine
**Solution**: Add "keras" task type or "keras_" model prefix detection
**Priority**: P0 - Blocker

### 2.2 Important Gaps (User Workflow)

#### GAP-5: Custom Architecture Specification
**Current State**: sklearn MLP only supports hidden_layer_sizes tuple
**User Needs**: Layer-by-layer specification (Dense, Dropout, custom activations)
**Solution**: JSON-based architecture builder
**Priority**: P1 - High

#### GAP-6: Training Configuration Control
**Current State**: Limited to max_iter parameter
**User Needs**: epochs, batch_size, kernel_initializer, verbose
**Solution**: Expose Keras training parameters in API
**Priority**: P1 - High

#### GAP-7: Kernel Initializer Variations
**Current State**: Not configurable
**User Needs**: random_normal, random_uniform, normal, uniform, glorot_uniform
**Solution**: Pass kernel_initializer to Dense layers
**Priority**: P1 - High

#### GAP-8: Multi-Model Training Workflow
**Current State**: Single model per train_model() call
**User Needs**: Train 8 variants in sequence
**Solution**: Batch training API or loop support
**Priority**: P2 - Medium

### 2.3 Optional Enhancements

#### GAP-9: Training History Tracking
**Feature**: Store epoch-by-epoch loss/accuracy
**Benefit**: Learning curves, convergence analysis
**Priority**: P3 - Nice to have

#### GAP-10: Keras Callbacks Support
**Feature**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
**Benefit**: Better training control
**Priority**: P3 - Nice to have

#### GAP-11: Position-Based Column Indexing
**Feature**: Support array[:, 2:16] style column selection
**Benefit**: Matches user's existing code style
**Priority**: P3 - Nice to have

---

## 3. Architecture Design

### 3.1 Component Overview

```
┌─────────────────────────────────────────────────────────┐
│                     ML Engine                           │
├─────────────────────────────────────────────────────────┤
│  train_model(task_type, model_type, ...)              │
│                                                         │
│  if model_type.startswith("keras_"):                  │
│      trainer = KerasNeuralNetworkTrainer               │
│  elif task_type == "neural_network":                   │
│      trainer = NeuralNetworkTrainer (sklearn)          │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│            KerasNeuralNetworkTrainer                    │
├─────────────────────────────────────────────────────────┤
│  - build_model(architecture_spec)                      │
│  - train(model, X, y, epochs, batch_size)             │
│  - get_model_summary()                                 │
│  - calculate_metrics()                                 │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│               ModelManager                              │
├─────────────────────────────────────────────────────────┤
│  save_model():                                         │
│    if model_type.startswith("keras_"):                │
│      - Save JSON architecture                          │
│      - Save H5 weights                                 │
│      - Save metadata.json                              │
│    else:                                               │
│      - Save .pkl (sklearn models)                      │
│                                                         │
│  load_model():                                         │
│    if .json + .h5 exist:                              │
│      - Load Keras model                                │
│    elif .pkl exists:                                   │
│      - Load sklearn model                              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Model Type Naming Convention

**Format**: `keras_{task}_{variant}`

**Supported Model Types**:
- `keras_binary_classification` - Binary classification (user's use case)
- `keras_multiclass_classification` - Multi-class classification
- `keras_regression` - Regression tasks

**Architecture Variants** (future):
- `keras_binary_simple` - Single hidden layer
- `keras_binary_deep` - Multiple hidden layers with dropout
- `keras_binary_custom` - User-defined architecture

---

## 4. Implementation Specifications

### 4.1 File Structure

```
src/engines/trainers/
├── keras_trainer.py              # NEW: Keras trainer implementation
├── neural_network_trainer.py     # EXISTING: sklearn MLP trainer
├── regression_trainer.py
├── classification_trainer.py
└── ml_base.py

src/engines/
├── ml_engine.py                  # MODIFY: Add Keras routing
├── model_manager.py              # MODIFY: Add Keras save/load
├── ml_config.py                  # MODIFY: Add Keras config section
└── ml_validators.py

tests/unit/
├── test_ml_keras.py              # NEW: Keras trainer tests
└── test_ml_keras_integration.py  # NEW: Integration tests

scripts/
└── test_keras_workflow.py        # NEW: User workflow validation
```

### 4.2 API Design

#### 4.2.1 Training API (User-Facing)

```python
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
import pandas as pd

# Initialize
config = MLEngineConfig.get_default()
engine = MLEngine(config)

# Load data
data = pd.read_csv('training_data.csv')

# Define architecture (JSON format)
architecture = {
    "layers": [
        {
            "type": "Dense",
            "units": 14,
            "activation": "relu",
            "kernel_initializer": "random_normal",
            "input_dim": 14
        },
        {
            "type": "Dense",
            "units": 1,
            "activation": "sigmoid",
            "kernel_initializer": "random_normal"
        }
    ],
    "compile": {
        "loss": "binary_crossentropy",
        "optimizer": "adam",
        "metrics": ["accuracy"]
    }
}

# Train single model
result = engine.train_model(
    data=data,
    task_type="classification",
    model_type="keras_binary_classification",
    target_column="target",
    feature_columns=["col1", "col2", ..., "col14"],
    user_id=12345,
    hyperparameters={
        "architecture": architecture,
        "epochs": 300,
        "batch_size": 70,
        "verbose": 1,
        "validation_split": 0.0  # No validation (train on 100%)
    }
)

print(f"Model ID: {result['model_id']}")
print(f"Accuracy: {result['metrics']['accuracy']}")
```

#### 4.2.2 Multi-Model Training Workflow

```python
# Train 8 variants (matching user's script)
variants = [
    {"initializer": "random_normal", "epochs": 300, "batch_size": 70},
    {"initializer": "random_normal", "epochs": 400, "batch_size": 90},
    {"initializer": "random_uniform", "epochs": 300, "batch_size": 70},
    {"initializer": "random_uniform", "epochs": 400, "batch_size": 90},
    {"initializer": "normal", "epochs": 300, "batch_size": 70},
    {"initializer": "normal", "epochs": 400, "batch_size": 90},
    {"initializer": "uniform", "epochs": 300, "batch_size": 70},
    {"initializer": "uniform", "epochs": 400, "batch_size": 90},
]

model_ids = []
for i, variant in enumerate(variants):
    # Update architecture with variant's initializer
    architecture["layers"][0]["kernel_initializer"] = variant["initializer"]
    architecture["layers"][1]["kernel_initializer"] = variant["initializer"]

    result = engine.train_model(
        data=data,
        task_type="classification",
        model_type="keras_binary_classification",
        target_column="target",
        feature_columns=feature_cols,
        user_id=12345,
        hyperparameters={
            "architecture": architecture,
            "epochs": variant["epochs"],
            "batch_size": variant["batch_size"]
        }
    )

    model_ids.append(result['model_id'])
    print(f"Model {i+1}/8: {result['model_id']}, Accuracy: {result['metrics']['accuracy']:.4f}")
```

### 4.3 Keras Trainer Implementation

**File**: `src/engines/trainers/keras_trainer.py`

**Key Methods**:

```python
class KerasNeuralNetworkTrainer(ModelTrainer):
    """Trainer for Keras Sequential models."""

    SUPPORTED_MODELS = [
        "keras_binary_classification",
        "keras_multiclass_classification",
        "keras_regression"
    ]

    def build_model_from_architecture(
        self,
        architecture: Dict[str, Any],
        n_features: int
    ) -> keras.Model:
        """
        Build Keras Sequential model from architecture specification.

        Args:
            architecture: Dict with "layers" and "compile" keys
            n_features: Number of input features

        Returns:
            Compiled Keras model
        """
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout

        model = Sequential()

        for i, layer_spec in enumerate(architecture["layers"]):
            layer_type = layer_spec["type"]

            if layer_type == "Dense":
                # First layer needs input_dim
                if i == 0:
                    model.add(Dense(
                        units=layer_spec["units"],
                        activation=layer_spec.get("activation", "relu"),
                        kernel_initializer=layer_spec.get("kernel_initializer", "glorot_uniform"),
                        input_dim=n_features
                    ))
                else:
                    model.add(Dense(
                        units=layer_spec["units"],
                        activation=layer_spec.get("activation", "relu"),
                        kernel_initializer=layer_spec.get("kernel_initializer", "glorot_uniform")
                    ))

            elif layer_type == "Dropout":
                model.add(Dropout(rate=layer_spec.get("rate", 0.5)))

        # Compile model
        compile_config = architecture.get("compile", {})
        model.compile(
            loss=compile_config.get("loss", "binary_crossentropy"),
            optimizer=compile_config.get("optimizer", "adam"),
            metrics=compile_config.get("metrics", ["accuracy"])
        )

        return model

    def train(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1,
        validation_split: float = 0.0
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train Keras model.

        Returns:
            Tuple of (trained_model, training_history)
        """
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split
        )

        # Convert history to serializable format
        training_history = {
            "loss": [float(x) for x in history.history["loss"]],
            "accuracy": [float(x) for x in history.history.get("accuracy", [])],
        }

        if validation_split > 0:
            training_history["val_loss"] = [float(x) for x in history.history.get("val_loss", [])]
            training_history["val_accuracy"] = [float(x) for x in history.history.get("val_accuracy", [])]

        return model, training_history

    def calculate_metrics(
        self,
        model: Any,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """Calculate metrics for Keras model."""
        # Evaluate on test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)

        # Get predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()

        # Calculate additional metrics
        from sklearn.metrics import (
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix
        )

        return {
            "loss": float(test_loss),
            "accuracy": float(test_accuracy),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
        }
```

### 4.4 ModelManager Updates

**File**: `src/engines/model_manager.py`

**Modified Methods**:

```python
def save_model(
    self,
    user_id: int,
    model_id: str,
    model: Any,
    metadata: Dict[str, Any],
    scaler: Optional[Any] = None,
    feature_info: Optional[Dict[str, Any]] = None,
    model_type: str = "sklearn"  # NEW parameter
) -> None:
    """Save model with format detection."""

    model_dir = self.get_user_models_dir(user_id) / model_id
    model_dir.mkdir(parents=True, exist_ok=True)

    # Detect if Keras model
    is_keras = hasattr(model, 'to_json') and hasattr(model, 'save_weights')

    if is_keras:
        # Save Keras model
        # 1. Save architecture as JSON
        model_json = model.to_json()
        with open(model_dir / "model.json", "w") as json_file:
            json_file.write(model_json)

        # 2. Save weights as H5
        model.save_weights(str(model_dir / "model.h5"))

        # 3. Mark as Keras in metadata
        metadata["model_format"] = "keras"
    else:
        # Save sklearn model
        joblib.dump(model, model_dir / "model.pkl")
        metadata["model_format"] = "sklearn"

    # Save metadata (always JSON)
    metadata["model_id"] = model_id
    metadata["user_id"] = user_id
    metadata["created_at"] = datetime.utcnow().isoformat() + "Z"

    with open(model_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save scaler if provided
    if scaler is not None:
        joblib.dump(scaler, model_dir / "scaler.pkl")

    # Save feature info
    if feature_info is not None:
        with open(model_dir / "feature_names.json", "w") as f:
            json.dump(feature_info, f, indent=2)


def load_model(self, user_id: int, model_id: str) -> Dict[str, Any]:
    """Load model with automatic format detection."""

    model_dir = self.get_model_dir(user_id, model_id)

    # Load metadata first to check format
    with open(model_dir / "metadata.json", "r") as f:
        metadata = json.load(f)

    model_format = metadata.get("model_format", "sklearn")

    if model_format == "keras":
        # Load Keras model
        from tensorflow.keras.models import model_from_json

        # Load architecture
        with open(model_dir / "model.json", "r") as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)

        # Load weights
        model.load_weights(str(model_dir / "model.h5"))
    else:
        # Load sklearn model
        model = joblib.load(model_dir / "model.pkl")

    # Load scaler if exists
    scaler = None
    scaler_path = model_dir / "scaler.pkl"
    if scaler_path.exists():
        scaler = joblib.load(scaler_path)

    # Load feature info
    feature_info = {}
    feature_info_path = model_dir / "feature_names.json"
    if feature_info_path.exists():
        with open(feature_info_path, "r") as f:
            feature_info = json.load(f)

    return {
        "model": model,
        "metadata": metadata,
        "scaler": scaler,
        "feature_info": feature_info
    }
```

### 4.5 ML Engine Integration

**File**: `src/engines/ml_engine.py`

**Modified `get_trainer()` method**:

```python
def get_trainer(self, task_type: str, model_type: str) -> Any:
    """
    Get appropriate trainer for task and model type.

    Args:
        task_type: Task type (regression, classification, neural_network)
        model_type: Model type (e.g., linear, keras_binary_classification)

    Returns:
        Trainer instance
    """
    # Check if Keras model (prefix-based detection)
    if model_type.startswith("keras_"):
        from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
        return KerasNeuralNetworkTrainer(self.config)

    # Otherwise use existing trainers
    if task_type == "regression":
        return self.trainers["regression"]
    elif task_type == "classification":
        return self.trainers["classification"]
    elif task_type == "neural_network":
        return self.trainers["neural_network"]
    else:
        raise ValidationError(
            f"Unknown task type: '{task_type}'",
            field="task_type",
            value=task_type
        )
```

**Modified `train_model()` method**:

```python
def train_model(
    self,
    data: pd.DataFrame,
    task_type: str,
    model_type: str,
    target_column: str,
    feature_columns: List[str],
    user_id: int,
    hyperparameters: Optional[Dict[str, Any]] = None,
    preprocessing_config: Optional[Dict[str, Any]] = None,
    test_size: Optional[float] = None,
    validation_type: str = "hold_out",
    cv_folds: Optional[int] = None
) -> Dict[str, Any]:
    """Train model (supports both sklearn and Keras)."""

    # ... existing preprocessing code ...

    # Get appropriate trainer (now supports Keras)
    trainer = self.get_trainer(task_type, model_type)

    # Check if Keras model
    is_keras = model_type.startswith("keras_")

    if is_keras:
        # Keras-specific training path
        architecture = hyperparameters.get("architecture", {})
        epochs = hyperparameters.get("epochs", 100)
        batch_size = hyperparameters.get("batch_size", 32)
        verbose = hyperparameters.get("verbose", 1)
        validation_split = hyperparameters.get("validation_split", 0.0)

        # Build model from architecture
        model = trainer.build_model_from_architecture(
            architecture,
            n_features=len(feature_columns)
        )

        # Train model (returns model and history)
        trained_model, training_history = trainer.train(
            model,
            X_train_scaled,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split
        )

        # Calculate metrics
        validation_results = trainer.calculate_metrics(
            trained_model,
            X_test_scaled,
            y_test
        )

        # Add training history to metadata
        metadata["training_history"] = training_history
    else:
        # Existing sklearn training path
        model = trainer.get_model_instance(model_type, hyperparameters)
        trained_model = trainer.train(model, X_train_scaled, y_train)
        validation_results = trainer.validate_model(
            trained_model,
            X_test_scaled,
            y_test
        )

    # ... rest of save logic (now detects Keras vs sklearn) ...
```

### 4.6 Configuration Updates

**File**: `config/config.yaml`

Add new section:

```yaml
ml_engine:
  # ... existing sklearn config ...

  # Keras-specific configuration
  keras:
    enabled: true
    default_epochs: 100
    default_batch_size: 32
    default_verbose: 1
    default_validation_split: 0.0

    # Supported initializers
    supported_initializers:
      - glorot_uniform
      - glorot_normal
      - he_uniform
      - he_normal
      - random_normal
      - random_uniform
      - normal
      - uniform

    # Default architecture for binary classification
    default_binary_architecture:
      layers:
        - type: Dense
          units: 64
          activation: relu
          kernel_initializer: glorot_uniform
        - type: Dense
          units: 1
          activation: sigmoid
          kernel_initializer: glorot_uniform
      compile:
        loss: binary_crossentropy
        optimizer: adam
        metrics: [accuracy]
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**File**: `tests/unit/test_ml_keras.py`

**Test Cases**:

```python
class TestKerasTrainer:
    """Test Keras trainer functionality."""

    def test_build_model_simple_architecture(self):
        """Test building simple 2-layer model."""
        architecture = {
            "layers": [
                {"type": "Dense", "units": 14, "activation": "relu"},
                {"type": "Dense", "units": 1, "activation": "sigmoid"}
            ],
            "compile": {
                "loss": "binary_crossentropy",
                "optimizer": "adam"
            }
        }

        trainer = KerasNeuralNetworkTrainer(config)
        model = trainer.build_model_from_architecture(architecture, n_features=14)

        assert len(model.layers) == 2
        assert model.layers[0].units == 14
        assert model.layers[1].units == 1

    def test_train_binary_classification(self):
        """Test training Keras binary classification model."""
        # Create synthetic data
        X_train = pd.DataFrame(np.random.randn(100, 14))
        y_train = pd.Series(np.random.randint(0, 2, 100))

        architecture = {...}
        trainer = KerasNeuralNetworkTrainer(config)
        model = trainer.build_model_from_architecture(architecture, 14)

        trained_model, history = trainer.train(
            model, X_train, y_train,
            epochs=10, batch_size=32, verbose=0
        )

        assert "loss" in history
        assert "accuracy" in history
        assert len(history["loss"]) == 10

    def test_kernel_initializers(self):
        """Test different kernel initializers."""
        initializers = ["random_normal", "random_uniform", "normal", "uniform"]

        for init in initializers:
            architecture = {
                "layers": [
                    {"type": "Dense", "units": 14, "kernel_initializer": init},
                    {"type": "Dense", "units": 1, "activation": "sigmoid"}
                ],
                "compile": {"loss": "binary_crossentropy", "optimizer": "adam"}
            }

            trainer = KerasNeuralNetworkTrainer(config)
            model = trainer.build_model_from_architecture(architecture, 14)

            # Verify initializer was applied
            assert model.layers[0].kernel_initializer.__class__.__name__.lower().replace("_", "") == init.replace("_", "")

    def test_save_load_keras_model(self):
        """Test Keras model serialization."""
        # Train model
        trainer = KerasNeuralNetworkTrainer(config)
        model = trainer.build_model_from_architecture(architecture, 14)
        trained_model, _ = trainer.train(model, X_train, y_train, epochs=5)

        # Save
        manager = ModelManager(config)
        manager.save_model(
            user_id=12345,
            model_id="test_keras_model",
            model=trained_model,
            metadata={"model_type": "keras_binary_classification"},
            model_type="keras"
        )

        # Load
        loaded = manager.load_model(12345, "test_keras_model")

        assert loaded["metadata"]["model_format"] == "keras"
        assert loaded["model"] is not None

        # Verify predictions match
        orig_pred = trained_model.predict(X_test)
        loaded_pred = loaded["model"].predict(X_test)
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
```

### 5.2 Integration Tests

**File**: `tests/integration/test_ml_keras_integration.py`

```python
class TestKerasMLEngineIntegration:
    """Test full ML Engine workflow with Keras models."""

    def test_full_training_workflow(self):
        """Test complete train → save → load → predict cycle."""
        # Setup
        engine = MLEngine(MLEngineConfig.get_default())
        data = pd.DataFrame(np.random.randn(200, 15))
        data.columns = [f"feature_{i}" for i in range(14)] + ["target"]
        data["target"] = np.random.randint(0, 2, 200)

        # Train
        result = engine.train_model(
            data=data,
            task_type="classification",
            model_type="keras_binary_classification",
            target_column="target",
            feature_columns=[f"feature_{i}" for i in range(14)],
            user_id=99999,
            hyperparameters={
                "architecture": {...},
                "epochs": 50,
                "batch_size": 32
            }
        )

        assert "model_id" in result
        assert result["metrics"]["accuracy"] > 0.4

        # Predict
        new_data = pd.DataFrame(np.random.randn(10, 14))
        new_data.columns = [f"feature_{i}" for i in range(14)]

        predictions = engine.predict(
            user_id=99999,
            model_id=result["model_id"],
            data=new_data
        )

        assert len(predictions["predictions"]) == 10
        assert "probabilities" in predictions

    def test_multi_model_training_workflow(self):
        """Test training 8 variants like user's script."""
        engine = MLEngine(MLEngineConfig.get_default())

        variants = [
            {"initializer": "random_normal", "epochs": 50, "batch_size": 32},
            {"initializer": "random_uniform", "epochs": 50, "batch_size": 32},
            # ... 6 more variants
        ]

        model_ids = []
        for variant in variants:
            result = engine.train_model(...)
            model_ids.append(result["model_id"])

        assert len(model_ids) == 8

        # Verify all models saved
        models = engine.list_models(user_id=99999)
        assert len(models) >= 8
```

### 5.3 Workflow Validation Script

**File**: `scripts/test_keras_workflow.py`

Replicates user's exact workflow:

```python
#!/usr/bin/env python3
"""
Validate Keras workflow matches user's original script.

This script replicates the workflow from:
test_data/[ML] Fattor Response Model 1 - NN 14var CSV Training v1.py
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig

def main():
    print("=" * 60)
    print("KERAS WORKFLOW VALIDATION")
    print("Replicating user's 8-variant training")
    print("=" * 60)

    # Load data (simulated - user provides real CSV)
    np.random.seed(7)
    data = pd.DataFrame(np.random.randn(1000, 15))
    data.columns = [f"feature_{i}" for i in range(14)] + ["target"]
    data["target"] = np.random.randint(0, 2, 1000)

    # Initialize engine
    config = MLEngineConfig.get_default()
    engine = MLEngine(config)

    # Define base architecture
    base_architecture = {
        "layers": [
            {
                "type": "Dense",
                "units": 14,
                "activation": "relu",
                "kernel_initializer": "random_normal"  # Will be updated per variant
            },
            {
                "type": "Dense",
                "units": 1,
                "activation": "sigmoid",
                "kernel_initializer": "random_normal"
            }
        ],
        "compile": {
            "loss": "binary_crossentropy",
            "optimizer": "adam",
            "metrics": ["accuracy"]
        }
    }

    # Training variants (matching user's script)
    variants = [
        {"name": "RES_P1", "init": "random_normal", "epochs": 300, "batch": 70},
        {"name": "RES_P2", "init": "random_normal", "epochs": 400, "batch": 90},
        {"name": "RES_P3", "init": "random_uniform", "epochs": 300, "batch": 70},
        {"name": "RES_P4", "init": "random_uniform", "epochs": 400, "batch": 90},
        {"name": "RES_P5", "init": "normal", "epochs": 300, "batch": 70},
        {"name": "RES_P6", "init": "normal", "epochs": 400, "batch": 90},
        {"name": "RES_P7", "init": "uniform", "epochs": 300, "batch": 70},
        {"name": "RES_P8", "init": "uniform", "epochs": 400, "batch": 90},
    ]

    results = []
    for i, variant in enumerate(variants, 1):
        print(f"\n[{i}/8] Training {variant['name']}...")
        print(f"  Initializer: {variant['init']}")
        print(f"  Epochs: {variant['epochs']}, Batch Size: {variant['batch']}")

        # Update architecture with variant's initializer
        architecture = base_architecture.copy()
        architecture["layers"][0]["kernel_initializer"] = variant["init"]
        architecture["layers"][1]["kernel_initializer"] = variant["init"]

        # Train model
        result = engine.train_model(
            data=data,
            task_type="classification",
            model_type="keras_binary_classification",
            target_column="target",
            feature_columns=[f"feature_{i}" for i in range(14)],
            user_id=12345,
            hyperparameters={
                "architecture": architecture,
                "epochs": variant["epochs"],
                "batch_size": variant["batch"],
                "verbose": 0,
                "validation_split": 0.0
            }
        )

        results.append({
            "name": variant["name"],
            "model_id": result["model_id"],
            "accuracy": result["metrics"]["accuracy"],
            "loss": result["metrics"]["loss"]
        })

        print(f"  ✓ Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  ✓ Loss: {result['metrics']['loss']:.4f}")
        print(f"  ✓ Saved as: {result['model_id']}")

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - Summary")
    print("=" * 60)

    for r in results:
        print(f"{r['name']:8} | Accuracy: {r['accuracy']:.4f} | Loss: {r['loss']:.4f}")

    print(f"\n✓ All 8 models saved successfully")
    print(f"✓ Workflow matches user's original script")

if __name__ == "__main__":
    main()
```

---

## 6. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)
**Goal**: Basic Keras support functional

**Tasks**:
1. ✅ Add TensorFlow to requirements.txt
2. ✅ Create `keras_trainer.py` with basic Sequential model support
3. ✅ Update ModelManager for Keras serialization (JSON + H5)
4. ✅ Add Keras routing to ML Engine
5. ✅ Basic unit tests for trainer

**Deliverables**:
- Can train simple Keras model through ML Engine
- Can save and load Keras models
- Tests pass for basic workflow

**Validation**: Run simple 1-model training and prediction

---

### Phase 2: Architecture & Configuration (Week 2)
**Goal**: Full architecture flexibility

**Tasks**:
1. ✅ Implement JSON architecture builder
2. ✅ Support all layer types (Dense, Dropout)
3. ✅ Add kernel initializer variations
4. ✅ Expose epochs, batch_size, verbose parameters
5. ✅ Add configuration section to config.yaml
6. ✅ Training history tracking

**Deliverables**:
- Can specify custom architectures
- All training parameters configurable
- Training history saved in metadata

**Validation**: Train model with custom architecture

---

### Phase 3: Multi-Model & Testing (Week 3)
**Goal**: Production-ready, fully tested

**Tasks**:
1. ✅ Multi-model training workflow
2. ✅ Comprehensive unit tests
3. ✅ Integration tests
4. ✅ Workflow validation script
5. ✅ Performance testing
6. ✅ Documentation updates

**Deliverables**:
- Can train 8 variants like user's script
- 100% test coverage for Keras components
- Performance validated (training time, memory)
- User documentation complete

**Validation**: Run full 8-variant workflow matching user's script

---

## 7. Dependencies & Requirements

### 7.1 Python Package Updates

**requirements.txt additions**:
```
tensorflow>=2.12.0,<2.16.0  # Keras included
# Note: TensorFlow 2.12+ includes Keras by default
# Supports Python 3.8-3.11
```

**Why TensorFlow 2.12+**:
- Stable Keras API
- Good compatibility with existing sklearn
- Not bleeding edge (avoid breaking changes)

### 7.2 System Requirements

**Minimum**:
- Python 3.8+
- 4GB RAM (for small models)
- No GPU required (CPU training acceptable for user's model)

**Recommended**:
- Python 3.10
- 8GB RAM
- GPU optional (would speed up training but not required)

---

## 8. Backward Compatibility

### 8.1 Existing Functionality Preserved

**No Breaking Changes**:
- All existing sklearn models continue to work
- `neural_network_trainer.py` (sklearn MLP) unchanged
- Model loading automatically detects format
- Test suite for sklearn models still passes

### 8.2 Migration Path

**Users can**:
- Keep using sklearn MLP models
- Switch to Keras for specific use cases
- Mix both in same system

**Model Format Detection**:
```python
# Automatic based on files present
if .json + .h5 exist → Keras model
elif .pkl exists → sklearn model
```

---

## 9. Performance Considerations

### 9.1 Training Performance

**User's Script** (8 models):
- Epochs: 50-400 per model
- Estimated total time: 30-60 minutes (CPU)

**Optimization Strategies**:
- Batch processing for multiple variants
- Optional GPU support (if available)
- Early stopping to reduce unnecessary epochs
- Caching of preprocessed data

### 9.2 Memory Management

**Model Size**:
- User's model: ~14 inputs × 14 hidden × 1 output ≈ 200 parameters
- Minimal memory footprint (<1MB per model)
- JSON + H5 storage efficient

**Scaling**:
- Can handle 100+ models per user
- Existing size limits apply (100MB per model max)

---

## 10. Risk Assessment & Mitigation

### 10.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| TensorFlow dependency conflicts | Medium | High | Pin version range, test compatibility |
| Keras API changes | Low | Medium | Use stable TF 2.12, avoid deprecated APIs |
| Serialization failures | Medium | High | Extensive save/load testing, error handling |
| Performance issues | Low | Medium | Performance benchmarks, optimization |

### 10.2 User Impact Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Breaking existing workflows | Low | Critical | Maintain backward compatibility |
| Complex API adoption | Medium | Medium | Clear documentation, examples |
| Training too slow | Low | Medium | Optimize batch sizes, early stopping |

---

## 11. Success Criteria

### 11.1 Functional Requirements

✅ **Must Have**:
1. Train Keras Sequential models through ML Engine
2. Save models as JSON + H5
3. Load and predict with saved Keras models
4. Support 8-variant training workflow
5. All training parameters configurable (epochs, batch_size, initializer)

✅ **Should Have**:
6. Training history tracking
7. Custom architecture specification
8. Dropout layer support

🔲 **Nice to Have**:
9. Keras callbacks support (EarlyStopping, etc.)
10. GPU acceleration
11. Learning curve visualization

### 11.2 Quality Requirements

- ✅ 100% test coverage for new Keras components
- ✅ No regression in existing sklearn functionality
- ✅ Performance within 2x of user's original script
- ✅ Memory usage < 100MB per model
- ✅ Save/load reliability 100%

### 11.3 Validation Criteria

**Script**: `scripts/test_keras_workflow.py` must:
1. Train 8 models with different hyperparameters
2. Save all models successfully
3. Load and predict with all models
4. Complete in < 90 minutes (CPU)
5. Achieve accuracy > 50% on synthetic data

---

## 12. Timeline & Milestones

```
Week 1: Core Infrastructure
├─ Day 1-2: Dependencies, KerasTrainer skeleton
├─ Day 3-4: ModelManager updates, save/load
├─ Day 5: ML Engine routing, basic tests
└─ Milestone 1: Single Keras model trains successfully

Week 2: Architecture & Configuration
├─ Day 1-2: Architecture builder, layer types
├─ Day 3: Training parameters, initializers
├─ Day 4: Config file updates, defaults
├─ Day 5: Training history, metadata
└─ Milestone 2: Custom architectures working

Week 3: Testing & Validation
├─ Day 1-2: Unit tests, integration tests
├─ Day 3: Workflow validation script
├─ Day 4: Performance testing, optimization
├─ Day 5: Documentation, review
└─ Milestone 3: Production-ready, fully tested

Total: ~3 weeks for complete implementation
```

---

## 13. Next Steps

### Immediate Actions

1. **Review & Approve Plan** - User validates approach
2. **Setup Development Branch** - `feature/keras-neural-network-integration`
3. **Install Dependencies** - Add TensorFlow to local environment
4. **Create Test Data** - Prepare sample CSV matching user's format

### Implementation Order

1. **Phase 1 First** - Get basic training working end-to-end
2. **Validate Early** - Test with user's actual data ASAP
3. **Iterate on Feedback** - Adjust based on user workflow testing

---

## 14. Open Questions

**For User**:
1. Do you have a sample CSV file we can use for testing?
2. What's the typical size of your training datasets (rows × columns)?
3. Are there other Keras features you need (LSTM, CNN, etc.) or just Dense layers?
4. Do you need GPU support or is CPU training acceptable?
5. Should we support loading your existing saved models (JSON + H5)?

**For Technical Review**:
1. Should we support TensorFlow 2.15+ or stick with 2.12-2.14 for stability?
2. Should architecture spec support arbitrary Keras layers or limit to Dense/Dropout?
3. Should we add model versioning for architecture changes?

---

## 15. References

**User's Script**: `test_data/[ML] Fattor Response Model 1 - NN 14var CSV Training v1.py`

**Existing Code**:
- `src/engines/trainers/neural_network_trainer.py` - sklearn MLP reference
- `src/engines/model_manager.py` - Existing save/load patterns
- `src/engines/ml_engine.py` - Training orchestration

**Documentation**:
- Keras Sequential API: https://keras.io/guides/sequential_model/
- TensorFlow 2.x Guide: https://www.tensorflow.org/guide
- Model serialization: https://keras.io/api/models/model_saving_apis/

---

## Appendix A: User's Original Script (Annotated)

```python
# User's workflow breakdown:

# 1. DATA LOADING
dataset10 = pandas.read_csv(path0)
array10 = dataset10.values
X_train = array10[:, 2:16]    # 14 features (columns 2-15)
Y_train = array10[:, 16]       # Target (column 16)

# 2. PREPROCESSING
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# 3. MODEL ARCHITECTURE (macro_model function)
model = Sequential()
model.add(Dense(14, input_dim=14, kernel_initializer=initializer, activation='relu'))
# Note: Dropout layers commented out in actual script
model.add(Dense(1, kernel_initializer=initializer, activation='sigmoid'))

# 4. COMPILE
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. TRAIN
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size)

# 6. EVALUATE
scores = model.evaluate(X_train, Y_train)
print(f"Accuracy: {scores[1] * 100:.2f}%")

# 7. SAVE
model_json = model.to_json()
with open(prediction_name, "w") as json_file:
    json_file.write(model_json)
model.save_weights(weights_name)

# 8. REPEAT 8 TIMES with different hyperparameters
# initializers: random_normal, random_uniform, normal, uniform
# epochs: 50, 150, 200, 300, 400
# batch_size: 20, 40, 50, 70, 90
```

---

**End of Implementation Plan**

This plan provides complete specifications for integrating Keras neural network support into the ML Engine, matching your existing workflow while maintaining backward compatibility with sklearn models.
