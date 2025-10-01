# Machine Learning Engine Implementation Plan

**Document Version**: 1.0
**Date**: 2025-10-01
**Status**: Design Complete - Ready for Implementation

## Executive Summary

This document provides a comprehensive design and implementation plan for the Machine Learning Engine component of the Statistical Modeling Agent. The ML Engine enables users to train, validate, and deploy machine learning models through natural language conversations in Telegram.

**Key Design Principles**:
- **Consistency**: Follow existing stats_engine pattern with script-based execution
- **Security**: Sandboxed execution, input validation, user isolation
- **Modularity**: Extensible architecture for adding new model types
- **User Experience**: Conversational training, clear results, model management

**Implementation Scope**:
- ~15 new files
- ~2,500-3,000 lines of code
- 14-day implementation timeline
- Target test coverage: >80%

---

## 1. Architecture Overview

### 1.1 System Integration

The ML Engine integrates into the existing pipeline architecture:

```
User Message (Telegram)
    ↓
handlers.py (routes message)
    ↓
parser.py (NL → TaskDefinition)
    ↓
orchestrator.py (determines task type)
    ↓
┌───────────────────┐
│   ML ENGINE       │ ← NEW COMPONENT
│                   │
│  ┌─────────────┐  │
│  │ MLEngine    │  │
│  └─────────────┘  │
│         ↓         │
│  ┌─────────────┐  │
│  │ Trainers    │  │
│  └─────────────┘  │
│         ↓         │
│  ┌─────────────┐  │
│  │ Scripts     │  │
│  └─────────────┘  │
└───────────────────┘
    ↓
executor.py (sandboxed execution)
    ↓
result_processor.py (format output)
    ↓
telegram_bot.py (send response)
```

### 1.2 Architectural Decision: Script-Based vs Direct Execution

**Decision**: Use script generation approach (consistent with stats_engine)

**Rationale**:
- ✅ Maintains architectural consistency
- ✅ Leverages existing sandbox infrastructure
- ✅ Resource limits already configured
- ✅ Easier audit and logging
- ✅ Natural separation of concerns

**Trade-offs Accepted**:
- More complex state management (solved with model persistence)
- Multi-step workflows require careful design
- Model objects must be serialized/deserialized

### 1.3 Component Architecture

```
ml_engine/
├── ml_engine.py           # Main MLEngine orchestrator
├── ml_base.py             # ModelTrainer base class
├── ml_config.py           # Configuration management
├── ml_exceptions.py       # ML-specific exceptions
├── ml_validators.py       # Input validation
├── ml_preprocessors.py    # Data preprocessing
├── model_manager.py       # Model persistence & loading
│
├── trainers/
│   ├── __init__.py
│   ├── regression_trainer.py
│   ├── classification_trainer.py
│   └── neural_network_trainer.py
│
└── templates/
    ├── ml_regression_template.py
    ├── ml_classification_template.py
    ├── ml_neural_network_template.py
    └── ml_prediction_template.py
```

---

## 2. Data Flow & Interfaces

### 2.1 TaskDefinition Interface

**Training Task**:
```python
@dataclass
class TaskDefinition:
    task_type: Literal["ml_train"]
    operation: str  # "train_model", "train_neural_network"
    parameters: dict[str, Any]  # See structure below
    data_source: Optional[DataSource]
    user_id: int
    conversation_id: str

# Training parameters structure
parameters = {
    "model_type": "random_forest",  # Required
    "target_column": "price",        # Required
    "feature_columns": ["age", "size", "location"],  # Required
    "test_size": 0.2,                # Optional, default: 0.2
    "hyperparameters": {              # Optional, use defaults
        "n_estimators": 100,
        "max_depth": 10
    },
    "validation": "cross_validation", # Optional: "hold_out" or "cross_validation"
    "cv_folds": 5,                   # Optional, default: 5
    "preprocessing": {                # Optional
        "scaling": "standard",        # "standard", "minmax", "robust", "none"
        "missing_strategy": "mean"    # "mean", "median", "drop"
    }
}
```

**Prediction Task**:
```python
@dataclass
class TaskDefinition:
    task_type: Literal["ml_score"]
    operation: str  # "predict"
    parameters: dict[str, Any]  # See structure below
    data_source: Optional[DataSource]  # New data to score
    user_id: int
    conversation_id: str

# Prediction parameters structure
parameters = {
    "model_id": "model_abc123",      # Required
    "return_probabilities": True      # Optional, for classification
}
```

### 2.2 Script Result Interface

```python
@dataclass
class MLTrainingResult:
    success: bool
    model_id: Optional[str]           # Generated UUID
    metrics: dict[str, float]         # Train and test metrics
    training_time: float              # Seconds
    model_info: dict[str, Any]        # Model type, features, etc.
    error: Optional[str]              # Error message if failed

@dataclass
class MLPredictionResult:
    success: bool
    predictions: list[Any]            # Predicted values or classes
    probabilities: Optional[list[list[float]]]  # For classification
    feature_importance: Optional[dict[str, float]]
    model_info: dict[str, Any]
    error: Optional[str]
```

---

## 3. Core Components Design

### 3.1 MLEngine Main Class

**Location**: `src/engines/ml_engine.py`

```python
from typing import Any, Dict, List, Optional
import pandas as pd
from pathlib import Path

from src.core.parser import TaskDefinition
from src.engines.ml_config import MLEngineConfig
from src.generators.script_generator import ScriptGenerator
from src.execution.executor import SandboxExecutor
from src.utils.exceptions import ValidationError, ModelNotFoundError

class MLEngine:
    """
    Machine learning engine for training and prediction.

    Responsibilities:
    - Orchestrate ML training workflows
    - Generate and execute training scripts
    - Manage model persistence
    - Execute predictions with trained models
    - Provide model management (list, delete)
    """

    def __init__(self, config: MLEngineConfig):
        """Initialize ML engine with configuration."""
        self.config = config
        self.models_dir = config.models_dir
        self.script_generator = ScriptGenerator()
        self.executor = SandboxExecutor()

        # Initialize trainers
        from src.engines.trainers import (
            RegressionTrainer,
            ClassificationTrainer,
            NeuralNetworkTrainer
        )

        self.trainers = {
            "regression": RegressionTrainer(config),
            "classification": ClassificationTrainer(config),
            "neural_network": NeuralNetworkTrainer(config) if TENSORFLOW_AVAILABLE else None
        }

    async def train_model(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.

        Args:
            task: Task definition with model type and parameters
            data: Training data as pandas DataFrame

        Returns:
            Dictionary with training results and model ID

        Raises:
            ValidationError: Invalid inputs or insufficient data
            ExecutionError: Training script failed
        """
        # 1. Validate inputs
        self._validate_training_task(task, data)

        # 2. Determine trainer type
        trainer_type = self._determine_trainer_type(task.parameters["model_type"])

        # 3. Generate training script
        script = self.script_generator.generate_training_script(task)

        # 4. Execute training in sandbox
        result = await self.executor.run_sandboxed(
            script,
            {"dataframe": data.to_dict()},
            timeout=self.config.max_training_time
        )

        # 5. Parse and return results
        return self._parse_training_results(result)

    async def predict(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Generate predictions using trained model.

        Args:
            task: Task definition with model_id
            data: Input data for prediction

        Returns:
            Dictionary with predictions and metadata

        Raises:
            ModelNotFoundError: Model doesn't exist
            ValidationError: Input data incompatible with model
        """
        # 1. Validate model exists
        model_id = task.parameters["model_id"]
        self._validate_model_exists(model_id, task.user_id)

        # 2. Load model metadata
        metadata = self._load_model_metadata(model_id, task.user_id)

        # 3. Validate input data matches model
        self._validate_prediction_data(data, metadata)

        # 4. Generate prediction script
        script = self.script_generator.generate_prediction_script(task)

        # 5. Execute prediction
        result = await self.executor.run_sandboxed(
            script,
            {
                "dataframe": data.to_dict(),
                "model_path": str(self._get_model_path(model_id, task.user_id))
            },
            timeout=30
        )

        # 6. Parse and return predictions
        return self._parse_prediction_results(result)

    def list_models(self, user_id: int) -> List[Dict[str, Any]]:
        """
        List all models for a user.

        Args:
            user_id: User identifier

        Returns:
            List of model metadata dictionaries
        """
        user_dir = self.models_dir / f"user_{user_id}"
        if not user_dir.exists():
            return []

        models = []
        for model_dir in user_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    models.append(json.load(f))

        return sorted(models, key=lambda x: x["created_at"], reverse=True)

    def delete_model(self, model_id: str, user_id: int) -> bool:
        """
        Delete a trained model.

        Args:
            model_id: Model identifier
            user_id: User identifier (for ownership verification)

        Returns:
            True if deleted, False if not found
        """
        model_path = self._get_model_path(model_id, user_id)
        if model_path.exists():
            shutil.rmtree(model_path)
            return True
        return False

    # Private methods

    def _validate_training_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> None:
        """Validate training task inputs."""
        # Implementation in validators module
        pass

    def _determine_trainer_type(self, model_type: str) -> str:
        """Map model type to trainer category."""
        regression_models = ["linear", "ridge", "lasso", "elasticnet", "polynomial"]
        classification_models = ["logistic", "svm", "random_forest", "gradient_boosting"]

        if model_type in regression_models:
            return "regression"
        elif model_type in classification_models:
            return "classification"
        elif model_type == "neural_network":
            return "neural_network"
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _parse_training_results(self, result: dict) -> Dict[str, Any]:
        """Parse script execution results."""
        # Implementation details
        pass

    def _validate_model_exists(self, model_id: str, user_id: int) -> None:
        """Check model exists and user owns it."""
        # Implementation in validators module
        pass

    def _load_model_metadata(self, model_id: str, user_id: int) -> dict:
        """Load model metadata from disk."""
        # Implementation in model_manager module
        pass

    def _validate_prediction_data(
        self,
        data: pd.DataFrame,
        metadata: dict
    ) -> None:
        """Validate prediction data matches training data schema."""
        # Implementation in validators module
        pass

    def _get_model_path(self, model_id: str, user_id: int) -> Path:
        """Get path to model directory."""
        return self.models_dir / f"user_{user_id}" / model_id

    def _parse_prediction_results(self, result: dict) -> Dict[str, Any]:
        """Parse prediction script results."""
        # Implementation details
        pass
```

### 3.2 ModelTrainer Base Class

**Location**: `src/engines/ml_base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
import pandas as pd
from sklearn.model_selection import train_test_split

class ModelTrainer(ABC):
    """
    Base class for ML model trainers.

    Responsibilities:
    - Data preparation (split, validation)
    - Preprocessing coordination
    - Training orchestration
    - Metrics calculation
    - Model persistence
    """

    def __init__(self, config: MLEngineConfig):
        self.config = config

    @abstractmethod
    def get_model_instance(self, model_type: str, hyperparameters: dict):
        """Create model instance based on type and hyperparameters."""
        pass

    @abstractmethod
    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """Calculate appropriate metrics for model type."""
        pass

    def prepare_data(
        self,
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.

        Returns:
            X_train, X_test, y_train, y_test
        """
        X = data[feature_columns]
        y = data[target_column]

        return train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

    def preprocess_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        scaling: str = "standard"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """
        Preprocess features (scaling, encoding).

        Returns:
            X_train_processed, X_test_processed, scaler_object
        """
        # Implementation in preprocessors module
        pass

    def validate_model(
        self,
        model,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> dict:
        """
        Validate model on test set.

        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_test)

        # Get probabilities for classification if available
        y_proba = None
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)

        return self.calculate_metrics(y_test, y_pred, y_proba)

    def save_model(
        self,
        model,
        model_id: str,
        user_id: int,
        metadata: dict
    ) -> Path:
        """
        Save model with metadata.

        Returns:
            Path to saved model directory
        """
        # Implementation in model_manager module
        pass
```

### 3.3 Specialized Trainers

#### RegressionTrainer

**Location**: `src/engines/trainers/regression_trainer.py`

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)

class RegressionTrainer(ModelTrainer):
    """Trainer for regression models."""

    def get_model_instance(self, model_type: str, hyperparameters: dict):
        """Create regression model instance."""
        models = {
            "linear": LinearRegression(),
            "ridge": Ridge(**hyperparameters),
            "lasso": Lasso(**hyperparameters),
            "elasticnet": ElasticNet(**hyperparameters),
            "polynomial": Pipeline([
                ('poly', PolynomialFeatures(degree=hyperparameters.get('degree', 2))),
                ('linear', LinearRegression())
            ])
        }

        if model_type not in models:
            raise ValueError(f"Unknown regression model type: {model_type}")

        return models[model_type]

    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """Calculate regression metrics."""
        mse = mean_squared_error(y_true, y_pred)

        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred)
        }
```

#### ClassificationTrainer

**Location**: `src/engines/trainers/classification_trainer.py`

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

class ClassificationTrainer(ModelTrainer):
    """Trainer for classification models."""

    def get_model_instance(self, model_type: str, hyperparameters: dict):
        """Create classification model instance."""
        models = {
            "logistic": LogisticRegression(max_iter=1000, **hyperparameters),
            "svm": SVC(probability=True, **hyperparameters),
            "random_forest": RandomForestClassifier(**hyperparameters),
            "gradient_boosting": GradientBoostingClassifier(**hyperparameters)
        }

        if model_type not in models:
            raise ValueError(f"Unknown classification model type: {model_type}")

        return models[model_type]

    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """Calculate classification metrics."""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_true, y_pred, average='weighted', zero_division=0),
            "f1": f1_score(y_true, y_pred, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
        }

        # Add ROC-AUC for binary classification
        if y_proba is not None and len(np.unique(y_true)) == 2:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except Exception:
                pass  # Skip if calculation fails

        return metrics
```

#### NeuralNetworkTrainer

**Location**: `src/engines/trainers/neural_network_trainer.py`

```python
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

class NeuralNetworkTrainer(ModelTrainer):
    """Trainer for neural network models."""

    def __init__(self, config: MLEngineConfig):
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow not available for neural network training")
        super().__init__(config)

    def create_regression_nn(
        self,
        input_dim: int,
        hidden_layers: list[int] = [64, 32],
        dropout: float = 0.2
    ):
        """Create regression neural network."""
        model = Sequential()

        # Input layer
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(dropout))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout))

        # Output layer
        model.add(Dense(1))

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def create_classification_nn(
        self,
        input_dim: int,
        num_classes: int,
        hidden_layers: list[int] = [64, 32],
        dropout: float = 0.3
    ):
        """Create classification neural network."""
        model = Sequential()

        # Input layer
        model.add(Dense(hidden_layers[0], activation='relu', input_dim=input_dim))
        model.add(Dropout(dropout))

        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(Dense(units, activation='relu'))
            model.add(Dropout(dropout))

        # Output layer
        activation = 'sigmoid' if num_classes == 2 else 'softmax'
        output_units = 1 if num_classes == 2 else num_classes
        model.add(Dense(output_units, activation=activation))

        loss = 'binary_crossentropy' if num_classes == 2 else 'sparse_categorical_crossentropy'
        model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
        return model

    def get_model_instance(self, model_type: str, hyperparameters: dict):
        """Create neural network instance - handled differently than sklearn."""
        # Neural networks are created dynamically based on data shape
        # This is called during script generation
        pass

    def calculate_metrics(self, y_true, y_pred, y_proba=None) -> dict:
        """Calculate metrics based on task type."""
        # Detect if regression or classification based on y_true
        if len(np.unique(y_true)) <= 10:  # Likely classification
            return ClassificationTrainer(self.config).calculate_metrics(
                y_true, y_pred, y_proba
            )
        else:  # Regression
            return RegressionTrainer(self.config).calculate_metrics(
                y_true, y_pred
            )
```

---

## 4. Script Templates

### 4.1 Regression Training Template

**Location**: `src/generators/templates/ml_regression_template.py`

```python
REGRESSION_TRAINING_TEMPLATE = """
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score
)
import time
import uuid

# Read configuration from stdin
config = json.loads(sys.stdin.read())
df = pd.DataFrame(config['dataframe'])

# Extract parameters
model_type = {model_type!r}
target_column = {target_column!r}
feature_columns = {feature_columns!r}
test_size = {test_size}
hyperparameters = {hyperparameters!r}
preprocessing_config = {preprocessing_config!r}
validation_type = {validation_type!r}
cv_folds = {cv_folds}
user_id = {user_id}

# Prepare data
X = df[feature_columns]
y = df[target_column]

# Handle missing values
missing_strategy = preprocessing_config.get('missing_strategy', 'mean')
if missing_strategy == 'mean':
    X = X.fillna(X.mean())
elif missing_strategy == 'median':
    X = X.fillna(X.median())
elif missing_strategy == 'drop':
    X = X.dropna()
    y = y.loc[X.index]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)

# Preprocessing - scaling
scaling_type = preprocessing_config.get('scaling', 'standard')
scaler = None
if scaling_type == 'standard':
    scaler = StandardScaler()
elif scaling_type == 'minmax':
    scaler = MinMaxScaler()
elif scaling_type == 'robust':
    scaler = RobustScaler()

if scaler:
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

# Create model
if model_type == 'linear':
    model = LinearRegression()
elif model_type == 'ridge':
    model = Ridge(**hyperparameters)
elif model_type == 'lasso':
    model = Lasso(**hyperparameters)
elif model_type == 'elasticnet':
    model = ElasticNet(**hyperparameters)
elif model_type == 'polynomial':
    degree = hyperparameters.get('degree', 2)
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=degree)),
        ('linear', LinearRegression())
    ])
else:
    raise ValueError(f"Unknown model type: {{model_type}}")

# Train model
start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Calculate metrics
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {{
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "explained_variance": float(explained_variance_score(y_true, y_pred))
    }}

train_metrics = calculate_metrics(y_train, y_train_pred)
test_metrics = calculate_metrics(y_test, y_test_pred)

# Cross-validation if requested
cv_metrics = None
if validation_type == 'cross_validation':
    cv_scores = cross_val_score(
        model, X_train, y_train,
        cv=cv_folds,
        scoring='r2'
    )
    cv_metrics = {{
        "mean_r2": float(cv_scores.mean()),
        "std_r2": float(cv_scores.std()),
        "scores": [float(s) for s in cv_scores]
    }}

# Generate model ID
model_id = f"model_{{uuid.uuid4().hex[:8]}}"

# Create model directory
model_dir = Path(f"models/user_{{user_id}}/{{model_id}}")
model_dir.mkdir(parents=True, exist_ok=True)

# Save model
joblib.dump(model, model_dir / "model.pkl")

# Save scaler if used
if scaler:
    joblib.dump(scaler, model_dir / "scaler.pkl")

# Save feature names
with open(model_dir / "feature_names.json", "w") as f:
    json.dump({{
        "features": feature_columns,
        "target": target_column
    }}, f)

# Save metadata
metadata = {{
    "model_id": model_id,
    "user_id": user_id,
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "model_type": model_type,
    "task_type": "regression",
    "target_column": target_column,
    "feature_columns": feature_columns,
    "preprocessing": {{
        "scaled": scaler is not None,
        "scaler_type": scaling_type if scaler else None,
        "missing_value_strategy": missing_strategy
    }},
    "hyperparameters": hyperparameters,
    "metrics": {{
        "train": train_metrics,
        "test": test_metrics,
        "cross_validation": cv_metrics
    }},
    "training_data_shape": list(X.shape),
    "training_time_seconds": training_time
}}

with open(model_dir / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Output results
results = {{
    "success": True,
    "model_id": model_id,
    "metrics": {{
        "train": train_metrics,
        "test": test_metrics,
        "cross_validation": cv_metrics
    }},
    "training_time": training_time,
    "model_info": {{
        "model_type": model_type,
        "features": feature_columns,
        "target": target_column
    }}
}}

print(json.dumps(results, indent=2))
"""
```

### 4.2 Classification Training Template

**Location**: `src/generators/templates/ml_classification_template.py`

Similar structure to regression template but with:
- Classification models (LogisticRegression, SVC, RandomForestClassifier, etc.)
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix generation
- Probability predictions
- Stratified splits

### 4.3 Neural Network Training Template

**Location**: `src/generators/templates/ml_neural_network_template.py`

Includes:
- TensorFlow/Keras model construction
- Layer configuration from hyperparameters
- EarlyStopping callback
- Training history tracking
- Model saving in SavedModel format
- Learning curves data

### 4.4 Prediction Template

**Location**: `src/generators/templates/ml_prediction_template.py`

```python
PREDICTION_TEMPLATE = """
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

# Read configuration
config = json.loads(sys.stdin.read())
df = pd.DataFrame(config['dataframe'])
model_path = Path(config['model_path'])

# Load metadata
with open(model_path / "metadata.json") as f:
    metadata = json.load(f)

# Load model
model = joblib.load(model_path / "model.pkl")

# Load scaler if exists
scaler = None
if (model_path / "scaler.pkl").exists():
    scaler = joblib.load(model_path / "scaler.pkl")

# Load feature names
with open(model_path / "feature_names.json") as f:
    feature_info = json.load(f)
    required_features = feature_info['features']

# Validate features
missing_features = set(required_features) - set(df.columns)
if missing_features:
    raise ValueError(f"Missing required features: {{missing_features}}")

# Prepare features
X = df[required_features]

# Apply same preprocessing as training
if scaler:
    X = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index
    )

# Generate predictions
predictions = model.predict(X)

# Get probabilities for classification
probabilities = None
if hasattr(model, 'predict_proba'):
    probabilities = model.predict_proba(X).tolist()

# Get feature importance if available
feature_importance = None
if hasattr(model, 'feature_importances_'):
    feature_importance = {{
        feat: float(imp)
        for feat, imp in zip(required_features, model.feature_importances_)
    }}
elif hasattr(model, 'coef_'):
    feature_importance = {{
        feat: float(coef)
        for feat, coef in zip(required_features, model.coef_.flatten())
    }}

# Format results
results = {{
    "success": True,
    "predictions": [float(p) if isinstance(p, (int, float, np.number)) else int(p) for p in predictions],
    "probabilities": probabilities,
    "feature_importance": feature_importance,
    "model_info": {{
        "model_id": metadata['model_id'],
        "model_type": metadata['model_type'],
        "trained_at": metadata['created_at']
    }}
}}

print(json.dumps(results, indent=2))
"""
```

---

## 5. Model Persistence

### 5.1 Storage Structure

```
models/
├── user_12345/
│   ├── model_abc12345/
│   │   ├── model.pkl              # Serialized model (joblib)
│   │   ├── metadata.json          # Training info, metrics
│   │   ├── scaler.pkl             # Feature scaler (if used)
│   │   └── feature_names.json     # Feature order/schema
│   │
│   └── model_def67890/
│       ├── model.pkl
│       ├── metadata.json
│       ├── scaler.pkl
│       └── feature_names.json
│
└── user_67890/
    └── model_xyz99999/
        └── ...
```

### 5.2 Metadata Schema

```json
{
  "model_id": "model_abc12345",
  "user_id": 12345,
  "created_at": "2025-10-01T10:30:00Z",
  "model_type": "random_forest",
  "task_type": "classification",
  "target_column": "outcome",
  "feature_columns": ["age", "income", "score"],
  "preprocessing": {
    "scaled": true,
    "scaler_type": "standard",
    "missing_value_strategy": "mean"
  },
  "hyperparameters": {
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 2
  },
  "metrics": {
    "train": {
      "accuracy": 0.95,
      "precision": 0.94,
      "recall": 0.93,
      "f1": 0.94,
      "roc_auc": 0.97
    },
    "test": {
      "accuracy": 0.89,
      "precision": 0.87,
      "recall": 0.88,
      "f1": 0.87,
      "roc_auc": 0.92
    },
    "cross_validation": {
      "mean_accuracy": 0.91,
      "std_accuracy": 0.03,
      "scores": [0.89, 0.92, 0.90, 0.93, 0.91]
    }
  },
  "training_data_shape": [1000, 4],
  "training_time_seconds": 2.5,
  "feature_importance": {
    "age": 0.15,
    "income": 0.45,
    "score": 0.40
  }
}
```

### 5.3 Model Manager

**Location**: `src/engines/model_manager.py`

```python
class ModelManager:
    """Handles model persistence and loading."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir

    def save_model(
        self,
        model,
        model_id: str,
        user_id: int,
        metadata: dict,
        scaler=None
    ) -> Path:
        """Save model with all artifacts."""
        model_dir = self.models_dir / f"user_{user_id}" / model_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        joblib.dump(model, model_dir / "model.pkl")

        # Save scaler if provided
        if scaler:
            joblib.dump(scaler, model_dir / "scaler.pkl")

        # Save metadata
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        return model_dir

    def load_model(self, model_id: str, user_id: int) -> Tuple[Any, dict, Any]:
        """Load model with metadata and scaler."""
        model_dir = self.models_dir / f"user_{user_id}" / model_id

        if not model_dir.exists():
            raise ModelNotFoundError(f"Model {model_id} not found")

        # Load model
        model = joblib.load(model_dir / "model.pkl")

        # Load metadata
        with open(model_dir / "metadata.json") as f:
            metadata = json.load(f)

        # Load scaler if exists
        scaler = None
        scaler_path = model_dir / "scaler.pkl"
        if scaler_path.exists():
            scaler = joblib.load(scaler_path)

        return model, metadata, scaler

    def delete_model(self, model_id: str, user_id: int) -> bool:
        """Delete model directory."""
        model_dir = self.models_dir / f"user_{user_id}" / model_id

        if model_dir.exists():
            shutil.rmtree(model_dir)
            return True
        return False

    def list_user_models(self, user_id: int) -> List[dict]:
        """List all models for a user."""
        user_dir = self.models_dir / f"user_{user_id}"

        if not user_dir.exists():
            return []

        models = []
        for model_dir in user_dir.iterdir():
            if not model_dir.is_dir():
                continue

            metadata_path = model_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    models.append(json.load(f))

        return sorted(models, key=lambda x: x["created_at"], reverse=True)
```

---

## 6. Validation & Preprocessing

### 6.1 Input Validators

**Location**: `src/engines/ml_validators.py`

```python
class MLValidators:
    """Input validation for ML operations."""

    @staticmethod
    def validate_training_data(
        data: pd.DataFrame,
        target_column: str,
        feature_columns: list[str],
        min_samples: int = 10
    ) -> None:
        """Validate training data."""
        # Check dataframe not empty
        if data.empty:
            raise ValidationError("Training data is empty")

        # Check sufficient samples
        if len(data) < min_samples:
            raise ValidationError(
                f"Insufficient training data: need at least {min_samples} samples, got {len(data)}"
            )

        # Check target column exists
        if target_column not in data.columns:
            raise ValidationError(f"Target column '{target_column}' not found in data")

        # Check feature columns exist
        missing_features = set(feature_columns) - set(data.columns)
        if missing_features:
            raise ValidationError(f"Feature columns not found: {missing_features}")

        # Check for target variability
        if data[target_column].nunique() == 1:
            raise ValidationError("Target column has no variance (all same value)")

    @staticmethod
    def validate_hyperparameters(
        model_type: str,
        hyperparameters: dict,
        allowed_ranges: dict
    ) -> None:
        """Validate hyperparameter values."""
        if model_type not in allowed_ranges:
            return  # No validation rules for this model type

        ranges = allowed_ranges[model_type]

        for param, value in hyperparameters.items():
            if param in ranges:
                min_val, max_val = ranges[param]
                if not (min_val <= value <= max_val):
                    raise ValidationError(
                        f"Hyperparameter '{param}' value {value} outside allowed range [{min_val}, {max_val}]"
                    )

    @staticmethod
    def validate_model_id(model_id: str) -> None:
        """Validate model ID format."""
        if not re.match(r'^model_[a-f0-9]{8}$', model_id):
            raise ValidationError("Invalid model ID format")

    @staticmethod
    def validate_prediction_data(
        data: pd.DataFrame,
        metadata: dict
    ) -> None:
        """Validate prediction data matches model schema."""
        required_features = metadata['feature_columns']

        # Check required features present
        missing_features = set(required_features) - set(data.columns)
        if missing_features:
            raise ValidationError(
                f"Prediction data missing required features: {missing_features}"
            )

        # Check data types match (basic validation)
        # Could be enhanced with more sophisticated type checking

        if data.empty:
            raise ValidationError("Prediction data is empty")
```

### 6.2 Data Preprocessors

**Location**: `src/engines/ml_preprocessors.py`

```python
class MLPreprocessors:
    """Data preprocessing utilities."""

    @staticmethod
    def handle_missing_values(
        X: pd.DataFrame,
        strategy: str = "mean"
    ) -> pd.DataFrame:
        """Handle missing values in features."""
        if strategy == "mean":
            return X.fillna(X.mean())
        elif strategy == "median":
            return X.fillna(X.median())
        elif strategy == "drop":
            return X.dropna()
        else:
            raise ValueError(f"Unknown missing value strategy: {strategy}")

    @staticmethod
    def scale_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        method: str = "standard"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """Scale numerical features."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }

        if method == "none":
            return X_train, X_test, None

        if method not in scalers:
            raise ValueError(f"Unknown scaling method: {method}")

        scaler = scalers[method]

        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_test_scaled, scaler

    @staticmethod
    def encode_categorical(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Encode categorical variables."""
        from sklearn.preprocessing import LabelEncoder

        encoders = {}
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                encoder = LabelEncoder()
                X_train_encoded[col] = encoder.fit_transform(X_train[col].astype(str))
                X_test_encoded[col] = encoder.transform(X_test[col].astype(str))
                encoders[col] = encoder

        return X_train_encoded, X_test_encoded, encoders
```

---

## 7. Integration with Existing Components

### 7.1 Orchestrator Updates

**File**: `src/core/orchestrator.py`

```python
class Orchestrator:
    def __init__(self, config):
        self.stats_engine = StatsEngine(config)
        self.ml_engine = MLEngine(config.ml_engine)  # NEW
        self.parser = Parser()

    async def execute_task(
        self,
        task: TaskDefinition,
        data: pd.DataFrame
    ) -> dict:
        """Route task to appropriate engine."""
        match task.task_type:
            case "stats":
                return await self.stats_engine.execute(task, data)

            case "ml_train":  # NEW
                return await self.ml_engine.train_model(task, data)

            case "ml_score":  # NEW
                return await self.ml_engine.predict(task, data)

            case _:
                raise ValueError(f"Unknown task type: {task.task_type}")
```

### 7.2 Script Generator Updates

**File**: `src/generators/script_generator.py`

```python
class ScriptGenerator:
    def __init__(self):
        self.template_dir = Path("src/generators/templates")

    def generate_training_script(self, task: TaskDefinition) -> str:
        """Generate ML training script."""
        model_type = task.parameters["model_type"]

        # Select template based on model type
        if model_type in ["linear", "ridge", "lasso", "elasticnet", "polynomial"]:
            template = self._load_template("ml_regression_template.py")
        elif model_type in ["logistic", "svm", "random_forest", "gradient_boosting"]:
            template = self._load_template("ml_classification_template.py")
        elif model_type == "neural_network":
            template = self._load_template("ml_neural_network_template.py")
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Render template with parameters
        return self._render_ml_template(template, task.parameters)

    def generate_prediction_script(self, task: TaskDefinition) -> str:
        """Generate prediction script."""
        template = self._load_template("ml_prediction_template.py")
        return self._render_ml_template(template, task.parameters)

    def _render_ml_template(self, template: str, parameters: dict) -> str:
        """Render ML template with parameters."""
        return template.format(
            model_type=parameters.get("model_type"),
            target_column=parameters.get("target_column"),
            feature_columns=parameters.get("feature_columns"),
            test_size=parameters.get("test_size", 0.2),
            hyperparameters=parameters.get("hyperparameters", {}),
            preprocessing_config=parameters.get("preprocessing", {}),
            validation_type=parameters.get("validation", "hold_out"),
            cv_folds=parameters.get("cv_folds", 5),
            user_id=parameters.get("user_id")
        )
```

### 7.3 Result Processor Updates

**File**: `src/processors/result_processor.py`

```python
class ResultProcessor:
    @staticmethod
    def format_training_results(results: dict) -> str:
        """Format ML training results for user."""
        if not results.get("success"):
            return f"❌ Training Failed\n\nError: {results.get('error', 'Unknown error')}"

        model_id = results["model_id"]
        metrics = results["metrics"]
        model_type = results["model_info"]["model_type"]

        output = f"🎯 Model Training Complete!\n\n"
        output += f"Model ID: {model_id}\n"
        output += f"Model Type: {model_type.replace('_', ' ').title()}\n\n"

        # Training metrics
        output += "📊 Training Metrics:\n"
        for metric, value in metrics["train"].items():
            if metric == "confusion_matrix":
                continue
            formatted_value = f"{value:.4f}"
            if metric in ["accuracy", "precision", "recall", "f1", "r2"]:
                formatted_value += f" ({value*100:.1f}%)"
            output += f"• {metric.upper()}: {formatted_value}\n"

        # Test metrics
        output += "\n📈 Test Metrics:\n"
        for metric, value in metrics["test"].items():
            if metric == "confusion_matrix":
                continue
            formatted_value = f"{value:.4f}"
            if metric in ["accuracy", "precision", "recall", "f1", "r2"]:
                formatted_value += f" ({value*100:.1f}%)"
            output += f"• {metric.upper()}: {formatted_value}\n"

        # Cross-validation if available
        if metrics.get("cross_validation"):
            cv = metrics["cross_validation"]
            output += "\n🔄 Cross-Validation:\n"
            for metric, value in cv.items():
                if metric != "scores":
                    output += f"• {metric}: {value:.4f}\n"

        output += f"\n✅ Model saved and ready for predictions!\n"
        output += f"\nUse this model ID for predictions:\n/predict {model_id}"

        return output

    @staticmethod
    def format_prediction_results(results: dict) -> str:
        """Format prediction results for user."""
        if not results.get("success"):
            return f"❌ Prediction Failed\n\nError: {results.get('error', 'Unknown error')}"

        model_info = results["model_info"]
        predictions = results["predictions"]
        probabilities = results.get("probabilities")

        output = f"🔮 Predictions Generated\n\n"
        output += f"Model: {model_info['model_id']} ({model_info['model_type']})\n\n"

        # Show first few predictions
        output += "Results (showing first 5):\n"
        for i, pred in enumerate(predictions[:5]):
            if probabilities:
                probs = probabilities[i]
                confidence = max(probs) * 100
                output += f"Row {i+1}: Class {pred} (confidence: {confidence:.0f}%)\n"
            else:
                output += f"Row {i+1}: {pred:.4f}\n"

        if len(predictions) > 5:
            output += f"... and {len(predictions) - 5} more\n"

        # Feature importance if available
        if results.get("feature_importance"):
            output += "\n📊 Feature Importance:\n"
            importance = sorted(
                results["feature_importance"].items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            for feat, imp in importance[:5]:
                output += f"• {feat}: {imp:.4f}\n"

        output += "\n📎 Full results available in conversation"

        return output
```

---

## 8. Configuration

### 8.1 Config File Updates

**File**: `config/config.yaml`

```yaml
# Existing configuration...

# ML Engine Configuration
ml_engine:
  # Model storage
  models_dir: "models"
  max_models_per_user: 50
  max_model_size_mb: 100

  # Training limits
  max_training_time_seconds: 300  # 5 minutes
  max_memory_mb: 2048
  min_training_samples: 10

  # Data preprocessing
  default_test_size: 0.2
  default_cv_folds: 5
  default_missing_strategy: "mean"  # mean, median, drop
  default_scaling: "standard"  # standard, minmax, robust, none

  # Model defaults
  default_hyperparameters:
    # Regression
    ridge:
      alpha: 1.0
    lasso:
      alpha: 1.0
    elasticnet:
      alpha: 1.0
      l1_ratio: 0.5
    polynomial:
      degree: 2

    # Classification
    logistic:
      max_iter: 1000
      C: 1.0
    svm:
      kernel: 'rbf'
      C: 1.0
    random_forest:
      n_estimators: 100
      max_depth: 10
      min_samples_split: 2
    gradient_boosting:
      n_estimators: 100
      learning_rate: 0.1
      max_depth: 3

    # Neural Networks
    neural_network:
      hidden_layers: [64, 32]
      epochs: 50
      batch_size: 32
      dropout: 0.2

  # Hyperparameter validation ranges
  hyperparameter_ranges:
    n_estimators: [10, 500]
    max_depth: [1, 50]
    learning_rate: [0.001, 0.5]
    epochs: [10, 200]
    batch_size: [8, 128]
    alpha: [0.0001, 10.0]
    C: [0.001, 100.0]
```

### 8.2 Config Loading

**File**: `src/engines/ml_config.py`

```python
from pathlib import Path
import yaml
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MLEngineConfig:
    """Configuration for ML Engine."""

    models_dir: Path
    max_models_per_user: int
    max_model_size_mb: int
    max_training_time: int
    max_memory_mb: int
    min_training_samples: int
    default_test_size: float
    default_cv_folds: int
    default_missing_strategy: str
    default_scaling: str
    default_hyperparameters: Dict[str, Dict[str, Any]]
    hyperparameter_ranges: Dict[str, list]

    @classmethod
    def from_yaml(cls, config_path: Path) -> "MLEngineConfig":
        """Load configuration from YAML file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        ml_config = config["ml_engine"]

        return cls(
            models_dir=Path(ml_config["models_dir"]),
            max_models_per_user=ml_config["max_models_per_user"],
            max_model_size_mb=ml_config["max_model_size_mb"],
            max_training_time=ml_config["max_training_time_seconds"],
            max_memory_mb=ml_config["max_memory_mb"],
            min_training_samples=ml_config["min_training_samples"],
            default_test_size=ml_config["default_test_size"],
            default_cv_folds=ml_config["default_cv_folds"],
            default_missing_strategy=ml_config["default_missing_strategy"],
            default_scaling=ml_config["default_scaling"],
            default_hyperparameters=ml_config["default_hyperparameters"],
            hyperparameter_ranges=ml_config["hyperparameter_ranges"]
        )
```

---

## 9. Dependencies

### 9.1 Requirements.txt Updates

```txt
# Existing dependencies
python-telegram-bot>=20.0
anthropic>=0.3.0
pandas>=2.0.0
numpy>=1.24.0

# ML-specific additions
scikit-learn>=1.3.0        # Core ML library
tensorflow>=2.13.0         # Neural networks
keras>=2.13.0              # High-level NN API
joblib>=1.3.0             # Model serialization
pyyaml>=6.0               # Config file parsing

# Optional but recommended
matplotlib>=3.7.0          # Visualization
seaborn>=0.12.0           # Statistical plots

# Development
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.7.0
mypy>=1.5.0
flake8>=6.1.0
```

---

## 10. Testing Strategy

### 10.1 Unit Tests

**File**: `tests/unit/test_ml_engine.py`

```python
import pytest
import pandas as pd
from src.engines.ml_engine import MLEngine
from src.core.parser import TaskDefinition

class TestMLEngine:
    """Test ML Engine functionality."""

    @pytest.fixture
    def ml_engine(self, tmp_path):
        """Create ML engine with temporary model storage."""
        config = MLEngineConfig(
            models_dir=tmp_path / "models",
            max_training_time=60,
            # ... other config
        )
        return MLEngine(config)

    @pytest.fixture
    def regression_data(self):
        """Sample regression dataset."""
        return pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'target': np.random.randn(100)
        })

    @pytest.fixture
    def classification_data(self):
        """Sample classification dataset."""
        return pd.DataFrame({
            'feature1': np.random.randn(200),
            'feature2': np.random.randn(200),
            'target': np.random.choice([0, 1], 200)
        })

    @pytest.mark.asyncio
    async def test_train_linear_regression(self, ml_engine, regression_data):
        """Test training linear regression model."""
        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={
                "model_type": "linear",
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "test_size": 0.2
            },
            data_source=None,
            user_id=12345,
            conversation_id="test_123"
        )

        result = await ml_engine.train_model(task, regression_data)

        assert result["success"] is True
        assert "model_id" in result
        assert "metrics" in result
        assert "train" in result["metrics"]
        assert "test" in result["metrics"]
        assert "r2" in result["metrics"]["test"]

    @pytest.mark.asyncio
    async def test_train_random_forest_classifier(
        self,
        ml_engine,
        classification_data
    ):
        """Test training random forest classifier."""
        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={
                "model_type": "random_forest",
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"],
                "test_size": 0.2,
                "hyperparameters": {
                    "n_estimators": 50,
                    "max_depth": 5
                }
            },
            data_source=None,
            user_id=12345,
            conversation_id="test_456"
        )

        result = await ml_engine.train_model(task, classification_data)

        assert result["success"] is True
        assert "accuracy" in result["metrics"]["test"]
        assert "f1" in result["metrics"]["test"]

    @pytest.mark.asyncio
    async def test_predict_with_trained_model(
        self,
        ml_engine,
        regression_data
    ):
        """Test prediction workflow."""
        # First train a model
        train_task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={
                "model_type": "linear",
                "target_column": "target",
                "feature_columns": ["feature1", "feature2"]
            },
            data_source=None,
            user_id=12345,
            conversation_id="test_789"
        )

        train_result = await ml_engine.train_model(train_task, regression_data)
        model_id = train_result["model_id"]

        # Now predict with new data
        predict_task = TaskDefinition(
            task_type="ml_score",
            operation="predict",
            parameters={
                "model_id": model_id
            },
            data_source=None,
            user_id=12345,
            conversation_id="test_789"
        )

        new_data = regression_data[["feature1", "feature2"]].head(10)
        predict_result = await ml_engine.predict(predict_task, new_data)

        assert predict_result["success"] is True
        assert len(predict_result["predictions"]) == 10

    def test_list_models(self, ml_engine):
        """Test listing user models."""
        models = ml_engine.list_models(user_id=12345)
        assert isinstance(models, list)

    def test_insufficient_training_data(self, ml_engine):
        """Test error on insufficient data."""
        small_data = pd.DataFrame({
            'feature1': [1, 2, 3],
            'target': [1, 2, 3]
        })

        task = TaskDefinition(
            task_type="ml_train",
            operation="train_model",
            parameters={
                "model_type": "linear",
                "target_column": "target",
                "feature_columns": ["feature1"]
            },
            data_source=None,
            user_id=12345,
            conversation_id="test_error"
        )

        with pytest.raises(ValidationError):
            await ml_engine.train_model(task, small_data)
```

### 10.2 Integration Tests

**File**: `tests/integration/test_ml_workflow.py`

```python
@pytest.mark.asyncio
class TestMLWorkflow:
    """Test complete ML workflows."""

    async def test_complete_regression_workflow(self, mock_bot, sample_csv):
        """Test end-to-end regression workflow."""
        # Upload data
        await mock_bot.send_document(sample_csv)

        # Start training
        response = await mock_bot.send_message(
            "Train a linear regression model to predict price using age and size"
        )

        assert "Training started" in response or "Model Training Complete" in response
        assert "model_" in response  # Model ID present
        assert "R2" in response or "r2" in response

    async def test_complete_classification_workflow(self, mock_bot, classification_csv):
        """Test end-to-end classification workflow."""
        # Upload data
        await mock_bot.send_document(classification_csv)

        # Start training
        response = await mock_bot.send_message(
            "Train a random forest to classify outcome using all features"
        )

        assert "Model Training Complete" in response
        assert "Accuracy" in response
        assert "F1" in response

    async def test_train_and_predict_workflow(self, mock_bot, sample_csv, new_data_csv):
        """Test training then prediction."""
        # Upload training data
        await mock_bot.send_document(sample_csv)

        # Train model
        train_response = await mock_bot.send_message(
            "Train a linear regression model for price prediction"
        )

        # Extract model ID from response
        model_id = extract_model_id(train_response)

        # Upload new data
        await mock_bot.send_document(new_data_csv)

        # Request predictions
        predict_response = await mock_bot.send_message(
            f"Use model {model_id} to make predictions"
        )

        assert "Predictions Generated" in predict_response
        assert "Row 1:" in predict_response
```

### 10.3 Test Coverage Requirements

- Unit test coverage: >80%
- All public methods must have tests
- Error paths must be tested
- Edge cases:
  - Empty dataframes
  - Single-row data
  - Missing values
  - Mismatched features in prediction
  - Invalid model IDs
  - Invalid hyperparameters

---

## 11. Security Considerations

### 11.1 Script Safety Validation

All generated scripts must be validated before execution:

```python
FORBIDDEN_PATTERNS = [
    r'__import__',
    r'exec\s*\(',
    r'eval\s*\(',
    r'open\s*\(',
    r'subprocess',
    r'os\.system',
    r'\.\./',  # Path traversal
    r'pickle\.loads',  # Arbitrary object deserialization
]

def validate_script_safety(script: str) -> bool:
    """Check script for dangerous operations."""
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, script):
            return False
    return True
```

### 11.2 Model Ownership Verification

```python
def validate_model_ownership(model_id: str, user_id: int) -> bool:
    """Verify user owns the model."""
    # Sanitize model_id
    if not re.match(r'^model_[a-f0-9]{8}$', model_id):
        raise ValidationError("Invalid model ID format")

    # Check path exists within user's directory
    model_path = Path(f"models/user_{user_id}/{model_id}")
    if not model_path.exists():
        raise ModelNotFoundError(f"Model {model_id} not found")

    return True
```

### 11.3 Resource Limits

- Maximum training time: 5 minutes (configurable)
- Maximum memory: 2GB
- Maximum model file size: 100MB
- Maximum models per user: 50
- Sandboxed execution environment

### 11.4 Input Sanitization

```python
def sanitize_column_name(name: str) -> str:
    """Sanitize column names for safe script generation."""
    # Only allow alphanumeric and underscore
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', name)

    # Ensure starts with letter or underscore
    if sanitized and sanitized[0].isdigit():
        sanitized = f'_{sanitized}'

    return sanitized
```

---

## 12. Implementation Roadmap

### Sprint 1: Foundation (Days 1-2)
**Goal**: Core infrastructure

- [ ] Create `ml_engine.py` with MLEngine class skeleton
- [ ] Create `ml_base.py` with ModelTrainer base class
- [ ] Create `ml_exceptions.py` with ML-specific exceptions
- [ ] Create `ml_config.py` with configuration management
- [ ] Update `config/config.yaml` with ML settings
- [ ] Update `requirements.txt` with ML dependencies
- [ ] Create `models/` directory structure

**Acceptance**: MLEngine can be instantiated with config

### Sprint 2: Regression Models (Days 3-4)
**Goal**: Complete regression training pipeline

- [ ] Implement `RegressionTrainer` class
- [ ] Create `ml_regression_template.py` script template
- [ ] Implement regression metrics calculation
- [ ] Implement data preprocessing for regression
- [ ] Write unit tests for regression training
- [ ] Test end-to-end regression workflow manually

**Acceptance**: Can train and validate linear/ridge/lasso models

### Sprint 3: Classification Models (Days 5-6)
**Goal**: Complete classification training pipeline

- [ ] Implement `ClassificationTrainer` class
- [ ] Create `ml_classification_template.py` script template
- [ ] Implement classification metrics calculation
- [ ] Handle class imbalance detection
- [ ] Write unit tests for classification training
- [ ] Test end-to-end classification workflow

**Acceptance**: Can train and validate logistic/RF/GBM models

### Sprint 4: Neural Networks (Days 7-8)
**Goal**: Neural network support

- [ ] Implement `NeuralNetworkTrainer` class
- [ ] Create `ml_neural_network_template.py` script template
- [ ] Implement NN training with early stopping
- [ ] Handle both regression and classification NNs
- [ ] Write unit tests for NN training
- [ ] Test end-to-end NN workflow

**Acceptance**: Can train basic neural networks for both tasks

### Sprint 5: Prediction & Model Management (Days 9-10)
**Goal**: Complete prediction and lifecycle management

- [ ] Implement prediction functionality in MLEngine
- [ ] Create `ml_prediction_template.py` script template
- [ ] Implement `ModelManager` for persistence
- [ ] Implement `list_models()` functionality
- [ ] Implement `delete_model()` functionality
- [ ] Write unit tests for prediction and management
- [ ] Test train → save → load → predict cycle

**Acceptance**: Can predict with saved models, list and delete models

### Sprint 6: Integration (Days 11-12)
**Goal**: Wire into existing system

- [ ] Update `orchestrator.py` with ML routing logic
- [ ] Update `script_generator.py` with ML template handling
- [ ] Update `result_processor.py` with ML formatting
- [ ] Update `parser.py` to recognize ML requests (if needed)
- [ ] Write integration tests for full workflows
- [ ] Test multi-step ML conversations
- [ ] Test with real Telegram bot

**Acceptance**: Complete workflows work through Telegram interface

### Sprint 7: Validation & Polish (Days 13-14)
**Goal**: Production-ready quality

- [ ] Implement comprehensive input validation
- [ ] Implement data preprocessing pipeline
- [ ] Add detailed error messages for common issues
- [ ] Performance testing and optimization
- [ ] Security audit of generated scripts
- [ ] Code review and refactoring
- [ ] Documentation completion
- [ ] Final integration testing

**Acceptance**: >80% test coverage, all security checks pass

---

## 13. Error Handling

### 13.1 Error Hierarchy

```python
# src/utils/ml_exceptions.py

class MLError(AgentError):
    """Base exception for ML operations."""
    pass

class DataValidationError(MLError):
    """Input data validation failures."""
    pass

class ModelNotFoundError(MLError):
    """Model doesn't exist or user doesn't own it."""
    pass

class TrainingError(MLError):
    """Model training failures."""
    pass

class PredictionError(MLError):
    """Prediction execution failures."""
    pass

class FeatureMismatchError(PredictionError):
    """Prediction data doesn't match model schema."""
    pass

class ConvergenceError(TrainingError):
    """Model failed to converge during training."""
    pass
```

### 13.2 User-Friendly Error Messages

```python
ERROR_MESSAGES = {
    "insufficient_data": "Not enough training data. Need at least {min_samples} samples, but only {actual_samples} provided.",
    "missing_column": "Column '{column}' not found in your data.",
    "no_variance": "Target column has no variance (all values are the same). Cannot train a model.",
    "model_not_found": "Model '{model_id}' not found. Use /list_models to see your available models.",
    "feature_mismatch": "Prediction data is missing required features: {missing_features}",
    "training_timeout": "Training took too long (>{timeout}s). Try using less data or a simpler model.",
}
```

---

## 14. Future Enhancements

### Phase 2 Features (Post-MVP):

1. **Hyperparameter Tuning**
   - Grid search
   - Random search
   - Bayesian optimization

2. **Model Ensembles**
   - Voting classifiers
   - Stacking
   - Blending

3. **Feature Engineering**
   - Automatic feature selection
   - Polynomial features
   - Feature interactions

4. **Model Interpretation**
   - SHAP values
   - LIME explanations
   - Partial dependence plots

5. **Advanced Validation**
   - Time series cross-validation
   - Stratified sampling
   - Custom validation splits

6. **Model Comparison**
   - Train multiple models in parallel
   - Automatic model selection
   - Performance dashboards

7. **Production Features**
   - Model versioning
   - A/B testing support
   - Monitoring and drift detection
   - Automated retraining

---

## 15. Summary

This implementation plan provides a comprehensive roadmap for building the ML Engine. Key highlights:

### Architectural Strengths
- ✅ Consistent with existing patterns
- ✅ Secure by design (sandboxing, validation)
- ✅ Modular and extensible
- ✅ Well-tested (>80% coverage target)

### Core Capabilities
- ✅ Regression (5 model types)
- ✅ Classification (4 model types)
- ✅ Neural Networks (customizable)
- ✅ Model persistence and management
- ✅ Comprehensive metrics and validation

### Implementation Scope
- **Timeline**: 14 days (7 sprints)
- **Files**: ~15 new files
- **Code**: ~2,500-3,000 LOC
- **Tests**: ~1,000 LOC

### Success Criteria
- [ ] Can train regression models
- [ ] Can train classification models
- [ ] Can train neural networks
- [ ] Can make predictions with saved models
- [ ] Models persist correctly with metadata
- [ ] All metrics calculate correctly
- [ ] Error handling works for invalid inputs
- [ ] Security validation prevents unsafe operations
- [ ] Test coverage >80%
- [ ] Integration with Telegram bot works

This plan is ready for implementation.
