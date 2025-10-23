# /models Command - Interactive Model Browser

**Feature**: Browse all available ML models, view details, variants, parameters, and recommended uses via Telegram bot.

**Status**: Planning Complete
**Effort**: 9-12 hours
**LOC**: ~980 lines
**Created**: 2025-10-22

---

## 1. Executive Summary

### Feature Requirements

**User Workflow:**
1. User enters `/models` command
2. Bot displays paginated list of ALL available ML models (13+ types)
3. User clicks a model from the list
4. Bot shows detailed model information:
   - Model variants (binary/multiclass/regression)
   - Hyperparameters and configuration options
   - Recommended use cases and task types
   - Back button to return to models list
5. Models list view includes "Cancel" button to exit command

### Design Goals

1. **Discovery**: Help users find the right model for their task
2. **Education**: Explain model capabilities and use cases
3. **Navigation**: Intuitive back/forward navigation with state persistence
4. **Scalability**: Support 20+ models with pagination
5. **Integration**: Seamless with existing /train workflow

---

## 2. Architecture Analysis

### 2.1 Current State Management

From `src/core/state_manager.py` analysis:

```python
class WorkflowType(Enum):
    ML_TRAINING = "ml_training"
    ML_PREDICTION = "ml_prediction"
    STATS_ANALYSIS = "stats_analysis"
    DATA_EXPLORATION = "data_exploration"
    SCORE_WORKFLOW = "score_workflow"
```

**NEW: Add models browser workflow**

```python
class WorkflowType(Enum):
    # ...existing...
    MODELS_BROWSER = "models_browser"  # NEW
```

### 2.2 State Flow Diagram

```
┌─────────────────────────────────────────────────────┐
│                    /models                          │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌────────────────────────────────────────────────────┐
│  VIEWING_MODEL_LIST                                │
│  • Display: 3-5 models per page                    │
│  • Pagination: Prev/Next buttons                   │
│  • Action: Cancel button                           │
└──────────┬─────────────────────────────────────────┘
           │
           │ (User clicks model)
           ▼
┌────────────────────────────────────────────────────┐
│  VIEWING_MODEL_DETAILS                             │
│  • Display: Model name, description                │
│  • Variants: Buttons for each variant              │
│  • Parameters: Hyperparameter documentation        │
│  • Use Cases: Recommended applications             │
│  • Action: Back button → VIEWING_MODEL_LIST        │
└────────────────────────────────────────────────────┘
```

### 2.3 Existing Handler Patterns

**Pattern 1: Inline Keyboards**
```python
# From ml_training_local_path.py:145-152
keyboard = [
    [InlineKeyboardButton("📤 Upload File", callback_data="data_source:telegram")],
    [InlineKeyboardButton("📂 Use Local Path", callback_data="data_source:local_path")],
    [InlineKeyboardButton("📋 Use Template", callback_data="data_source:template")]
]
reply_markup = InlineKeyboardMarkup(keyboard)
```

**Pattern 2: State Snapshots (Back Button Support)**
```python
# From ml_training_local_path.py:197-198
session.save_state_snapshot()
self.logger.debug("📸 State snapshot saved before transition to AWAITING_DATA")
```

**Pattern 3: Callback Routing**
```python
# From existing handlers
choice = query.data.split(":")[-1]  # "telegram" or "local_path"
```

### 2.4 Model Registration System

From `src/engines/ml_engine.py` analysis:

**Existing Trainers:**
```python
self.trainers = {
    "regression": RegressionTrainer(config),           # 5 models
    "classification": ClassificationTrainer(config),   # 6 models
    "neural_network": NeuralNetworkTrainer(config)     # 2 models
}
```

**Dynamic Trainer Loading:**
```python
# Lines 67-84: Prefix-based detection
if model_type and model_type.startswith("catboost_"):
    from src.engines.trainers.catboost_trainer import CatBoostTrainer
    return CatBoostTrainer(self.config)

if model_type and model_type.startswith("lightgbm_"):
    from src.engines.trainers.lightgbm_trainer import LightGBMTrainer
    return LightGBMTrainer(self.config)

if model_type and model_type.startswith("xgboost_"):
    from src.engines.trainers.xgboost_trainer import XGBoostTrainer
    return XGBoostTrainer(self.config)

if model_type and model_type.startswith("keras_"):
    from src.engines.trainers.keras_trainer import KerasNeuralNetworkTrainer
    return KerasNeuralNetworkTrainer(self.config)
```

**Supported Models Discovery:**
```python
# Each trainer class has:
SUPPORTED_MODELS = [
    "model_type_1",
    "model_type_2",
    ...
]
```

---

## 3. Model Catalog Design

### 3.1 Data Structure

**File:** `src/engines/model_catalog.py` (NEW - 400 lines)

```python
from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum


class TaskType(Enum):
    """ML task types."""
    REGRESSION = "regression"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTICLASS_CLASSIFICATION = "multiclass_classification"
    NEURAL_NETWORK = "neural_network"


class ModelCategory(Enum):
    """Model categories for organization."""
    LINEAR_MODELS = "linear_models"
    TREE_BASED = "tree_based"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORKS = "neural_networks"
    ENSEMBLE = "ensemble"
    BAYESIAN = "bayesian"


@dataclass
class ModelParameter:
    """Model hyperparameter documentation."""
    name: str
    display_name: str
    description: str
    default: Optional[any]
    range: Optional[str]
    example: Optional[str]


@dataclass
class ModelInfo:
    """Complete model metadata."""
    id: str                          # Technical ID (e.g., "linear", "xgboost_binary")
    display_name: str                 # User-friendly name
    category: ModelCategory           # Organization category
    task_type: TaskType               # Supported task
    icon: str                         # Emoji for display
    short_description: str            # One-line summary
    long_description: str             # Detailed explanation
    parameters: List[ModelParameter]  # Hyperparameters
    use_cases: List[str]             # Recommended applications
    strengths: List[str]             # Model advantages
    limitations: List[str]           # Model disadvantages
    requires_tuning: bool            # Whether hyperparameter tuning needed
    training_speed: str              # "Fast", "Medium", "Slow"
    prediction_speed: str            # "Fast", "Medium", "Slow"
    interpretability: str            # "High", "Medium", "Low"
    variants: Optional[List[str]]    # For multi-task models (XGBoost, etc.)


# Master catalog of all models
MODEL_CATALOG: Dict[str, ModelInfo] = {
    # ========================================
    # REGRESSION MODELS (5)
    # ========================================
    "linear": ModelInfo(
        id="linear",
        display_name="Linear Regression",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.REGRESSION,
        icon="📈",
        short_description="Fast, interpretable linear relationship modeling",
        long_description=(
            "Linear regression models the relationship between features and "
            "target as a linear equation. Best for problems with linear "
            "relationships and when interpretability is important."
        ),
        parameters=[
            ModelParameter(
                name="fit_intercept",
                display_name="Fit Intercept",
                description="Whether to calculate intercept",
                default=True,
                range="True/False",
                example="True"
            )
        ],
        use_cases=[
            "House price prediction",
            "Sales forecasting",
            "Simple trend analysis",
            "Feature relationship analysis"
        ],
        strengths=[
            "Highly interpretable coefficients",
            "Very fast training and prediction",
            "Works well with small datasets",
            "No hyperparameter tuning needed"
        ],
        limitations=[
            "Only captures linear relationships",
            "Sensitive to outliers",
            "Assumes feature independence",
            "Poor with complex patterns"
        ],
        requires_tuning=False,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="High",
        variants=None
    ),

    "ridge": ModelInfo(
        id="ridge",
        display_name="Ridge Regression (L2)",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.REGRESSION,
        icon="📉",
        short_description="Linear regression with L2 regularization",
        long_description=(
            "Ridge regression adds L2 penalty to reduce overfitting. "
            "Handles multicollinearity better than standard linear regression "
            "by shrinking coefficients."
        ),
        parameters=[
            ModelParameter(
                name="alpha",
                display_name="Regularization Strength",
                description="L2 penalty strength (higher = more regularization)",
                default=1.0,
                range="0.0 to 10.0+",
                example="1.0"
            )
        ],
        use_cases=[
            "High-dimensional data",
            "Correlated features",
            "Prevent overfitting",
            "Stable predictions"
        ],
        strengths=[
            "Handles multicollinearity",
            "Reduces overfitting",
            "Interpretable",
            "Fast training"
        ],
        limitations=[
            "Requires alpha tuning",
            "Still assumes linearity",
            "Shrinks all coefficients"
        ],
        requires_tuning=True,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="High",
        variants=None
    ),

    "lasso": ModelInfo(
        id="lasso",
        display_name="Lasso Regression (L1)",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.REGRESSION,
        icon="🎯",
        short_description="Linear regression with L1 regularization and feature selection",
        long_description=(
            "Lasso performs automatic feature selection by driving some "
            "coefficients to exactly zero. Useful when you want a sparse model "
            "with only the most important features."
        ),
        parameters=[
            ModelParameter(
                name="alpha",
                display_name="Regularization Strength",
                description="L1 penalty strength (higher = more sparsity)",
                default=1.0,
                range="0.0 to 10.0+",
                example="0.5"
            )
        ],
        use_cases=[
            "Feature selection",
            "High-dimensional sparse data",
            "Identifying key predictors",
            "Interpretable models"
        ],
        strengths=[
            "Automatic feature selection",
            "Handles high dimensions",
            "Interpretable",
            "Reduces overfitting"
        ],
        limitations=[
            "Unstable with correlated features",
            "Requires alpha tuning",
            "May drop important features"
        ],
        requires_tuning=True,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="High",
        variants=None
    ),

    "elasticnet": ModelInfo(
        id="elasticnet",
        display_name="ElasticNet (L1 + L2)",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.REGRESSION,
        icon="⚖️",
        short_description="Combines L1 and L2 regularization",
        long_description=(
            "ElasticNet combines Lasso (L1) and Ridge (L2) regularization. "
            "Balances feature selection with coefficient shrinkage. "
            "Best of both worlds for complex datasets."
        ),
        parameters=[
            ModelParameter(
                name="alpha",
                display_name="Overall Regularization",
                description="Total penalty strength",
                default=1.0,
                range="0.0 to 10.0+",
                example="1.0"
            ),
            ModelParameter(
                name="l1_ratio",
                display_name="L1 Ratio",
                description="Balance between L1 and L2 (0=Ridge, 1=Lasso)",
                default=0.5,
                range="0.0 to 1.0",
                example="0.5"
            )
        ],
        use_cases=[
            "Correlated features",
            "High-dimensional data",
            "Feature selection + stability",
            "Complex datasets"
        ],
        strengths=[
            "Handles correlated features",
            "Feature selection",
            "Flexible regularization",
            "Stable"
        ],
        limitations=[
            "Two hyperparameters to tune",
            "Slower than Ridge/Lasso",
            "More complex tuning"
        ],
        requires_tuning=True,
        training_speed="Medium",
        prediction_speed="Fast",
        interpretability="High",
        variants=None
    ),

    "polynomial": ModelInfo(
        id="polynomial",
        display_name="Polynomial Regression",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.REGRESSION,
        icon="📐",
        short_description="Linear regression with polynomial features",
        long_description=(
            "Creates polynomial features (x², x³, etc.) to capture non-linear "
            "relationships while keeping the model structure linear. "
            "Useful for curved patterns."
        ),
        parameters=[
            ModelParameter(
                name="degree",
                display_name="Polynomial Degree",
                description="Highest polynomial degree (2=quadratic, 3=cubic)",
                default=2,
                range="2 to 5",
                example="2"
            )
        ],
        use_cases=[
            "Curved relationships",
            "Non-linear patterns",
            "Limited data",
            "Interpretability needed"
        ],
        strengths=[
            "Captures non-linear patterns",
            "Still interpretable",
            "Fast training",
            "No complex algorithms"
        ],
        limitations=[
            "Risk of overfitting (high degree)",
            "Features multiply quickly",
            "Extrapolation issues",
            "Degree tuning needed"
        ],
        requires_tuning=True,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="Medium",
        variants=None
    ),

    # ========================================
    # CLASSIFICATION MODELS (6)
    # ========================================
    "logistic": ModelInfo(
        id="logistic",
        display_name="Logistic Regression",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🎲",
        short_description="Fast, interpretable binary classification",
        long_description=(
            "Logistic regression uses a logistic function to model probability "
            "of binary outcomes. Despite the name, it's a classification model. "
            "Highly interpretable with probability outputs."
        ),
        parameters=[
            ModelParameter(
                name="C",
                display_name="Regularization Strength",
                description="Inverse regularization (lower = more regularization)",
                default=1.0,
                range="0.01 to 100",
                example="1.0"
            )
        ],
        use_cases=[
            "Binary classification",
            "Probability predictions",
            "Medical diagnosis",
            "Fraud detection"
        ],
        strengths=[
            "Probability outputs",
            "Fast training",
            "Interpretable coefficients",
            "Works well with small data"
        ],
        limitations=[
            "Only binary classification",
            "Assumes linear boundaries",
            "Sensitive to outliers"
        ],
        requires_tuning=False,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="High",
        variants=None
    ),

    "decision_tree": ModelInfo(
        id="decision_tree",
        display_name="Decision Tree Classifier",
        category=ModelCategory.TREE_BASED,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🌳",
        short_description="Interpretable tree-based classification",
        long_description=(
            "Decision trees create if-then rules by splitting data based on "
            "feature values. Highly interpretable but prone to overfitting. "
            "Foundation for ensemble methods like Random Forest."
        ),
        parameters=[
            ModelParameter(
                name="max_depth",
                display_name="Maximum Tree Depth",
                description="Maximum depth to prevent overfitting",
                default=None,
                range="3 to 20",
                example="10"
            ),
            ModelParameter(
                name="min_samples_split",
                display_name="Min Samples to Split",
                description="Minimum samples required to split node",
                default=2,
                range="2 to 20",
                example="5"
            )
        ],
        use_cases=[
            "Interpretable models",
            "Non-linear patterns",
            "Mixed data types",
            "Feature importance"
        ],
        strengths=[
            "Highly interpretable",
            "Handles non-linear patterns",
            "No feature scaling needed",
            "Fast prediction"
        ],
        limitations=[
            "Prone to overfitting",
            "Unstable (small data changes)",
            "Biased to dominant classes"
        ],
        requires_tuning=True,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="High",
        variants=None
    ),

    "random_forest": ModelInfo(
        id="random_forest",
        display_name="Random Forest Classifier",
        category=ModelCategory.ENSEMBLE,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🌲",
        short_description="Ensemble of decision trees for robust classification",
        long_description=(
            "Random Forest builds multiple decision trees on random subsets "
            "of data and features, then averages predictions. Reduces overfitting "
            "and improves accuracy. Very popular for general-purpose classification."
        ),
        parameters=[
            ModelParameter(
                name="n_estimators",
                display_name="Number of Trees",
                description="Number of trees in the forest",
                default=100,
                range="50 to 500",
                example="100"
            ),
            ModelParameter(
                name="max_depth",
                display_name="Maximum Tree Depth",
                description="Maximum depth per tree",
                default=None,
                range="5 to 30",
                example="15"
            )
        ],
        use_cases=[
            "General classification",
            "Feature importance",
            "Imbalanced data",
            "Robust predictions"
        ],
        strengths=[
            "Reduces overfitting",
            "Feature importance",
            "Handles missing values",
            "Robust to outliers"
        ],
        limitations=[
            "Slower than single tree",
            "Less interpretable",
            "Memory intensive",
            "Requires tuning"
        ],
        requires_tuning=True,
        training_speed="Medium",
        prediction_speed="Medium",
        interpretability="Medium",
        variants=None
    ),

    "gradient_boosting": ModelInfo(
        id="gradient_boosting",
        display_name="Gradient Boosting Classifier",
        category=ModelCategory.GRADIENT_BOOSTING,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="⚡",
        short_description="Sequential tree ensemble for high accuracy",
        long_description=(
            "Gradient Boosting builds trees sequentially, each correcting "
            "errors of previous trees. Often achieves highest accuracy but "
            "requires careful tuning. Sklearn's implementation (not XGBoost/LightGBM)."
        ),
        parameters=[
            ModelParameter(
                name="n_estimators",
                display_name="Number of Trees",
                description="Number of boosting stages",
                default=100,
                range="50 to 300",
                example="100"
            ),
            ModelParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Shrinks contribution of each tree",
                default=0.1,
                range="0.01 to 0.3",
                example="0.1"
            )
        ],
        use_cases=[
            "High accuracy needed",
            "Complex patterns",
            "Structured data",
            "Competitions"
        ],
        strengths=[
            "High accuracy",
            "Feature importance",
            "Handles mixed data types",
            "Flexible loss functions"
        ],
        limitations=[
            "Slow training",
            "Requires extensive tuning",
            "Prone to overfitting",
            "Less interpretable"
        ],
        requires_tuning=True,
        training_speed="Slow",
        prediction_speed="Medium",
        interpretability="Low",
        variants=None
    ),

    "svm": ModelInfo(
        id="svm",
        display_name="Support Vector Machine",
        category=ModelCategory.LINEAR_MODELS,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🔍",
        short_description="Finds optimal decision boundary using support vectors",
        long_description=(
            "SVM finds the hyperplane that best separates classes by maximizing "
            "margin between support vectors. Works well with high-dimensional "
            "data and kernel trick for non-linear boundaries."
        ),
        parameters=[
            ModelParameter(
                name="C",
                display_name="Regularization",
                description="Penalty for misclassification",
                default=1.0,
                range="0.1 to 10",
                example="1.0"
            ),
            ModelParameter(
                name="kernel",
                display_name="Kernel Type",
                description="Type of kernel (linear, rbf, poly)",
                default="rbf",
                range="linear, rbf, poly, sigmoid",
                example="rbf"
            )
        ],
        use_cases=[
            "High-dimensional data",
            "Text classification",
            "Image classification",
            "Clear margin separation"
        ],
        strengths=[
            "Effective in high dimensions",
            "Memory efficient",
            "Versatile kernels",
            "Works with small datasets"
        ],
        limitations=[
            "Slow with large datasets",
            "No probability estimates (default)",
            "Sensitive to scaling",
            "Complex tuning"
        ],
        requires_tuning=True,
        training_speed="Slow",
        prediction_speed="Medium",
        interpretability="Low",
        variants=None
    ),

    "naive_bayes": ModelInfo(
        id="naive_bayes",
        display_name="Naive Bayes Classifier",
        category=ModelCategory.BAYESIAN,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="📊",
        short_description="Probabilistic classifier based on Bayes theorem",
        long_description=(
            "Naive Bayes applies Bayes' theorem with assumption of feature "
            "independence. Very fast training and prediction. Particularly "
            "effective for text classification and spam detection."
        ),
        parameters=[],
        use_cases=[
            "Text classification",
            "Spam detection",
            "Sentiment analysis",
            "Real-time classification"
        ],
        strengths=[
            "Very fast training",
            "Fast prediction",
            "Works with small data",
            "Probability outputs"
        ],
        limitations=[
            "Assumes feature independence",
            "Less accurate than complex models",
            "Sensitive to correlated features"
        ],
        requires_tuning=False,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="Medium",
        variants=None
    ),

    # ========================================
    # GRADIENT BOOSTING VARIANTS
    # ========================================
    "xgboost": ModelInfo(
        id="xgboost",
        display_name="XGBoost",
        category=ModelCategory.GRADIENT_BOOSTING,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🚀",
        short_description="Optimized gradient boosting (3 variants)",
        long_description=(
            "XGBoost is an optimized implementation of gradient boosting with "
            "regularization, parallel processing, and tree pruning. "
            "State-of-the-art performance for structured data."
        ),
        parameters=[
            ModelParameter(
                name="n_estimators",
                display_name="Number of Trees",
                description="Number of boosting rounds",
                default=100,
                range="50 to 500",
                example="100"
            ),
            ModelParameter(
                name="max_depth",
                display_name="Maximum Depth",
                description="Maximum tree depth",
                default=6,
                range="3 to 15",
                example="6"
            ),
            ModelParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Step size shrinkage",
                default=0.1,
                range="0.01 to 0.3",
                example="0.1"
            )
        ],
        use_cases=[
            "Competition-winning accuracy",
            "Structured/tabular data",
            "Imbalanced datasets",
            "Feature engineering"
        ],
        strengths=[
            "State-of-the-art accuracy",
            "Regularization built-in",
            "Parallel processing",
            "Handles missing values"
        ],
        limitations=[
            "Requires extensive tuning",
            "Slower than LightGBM",
            "Memory intensive",
            "Requires OpenMP on macOS"
        ],
        requires_tuning=True,
        training_speed="Medium",
        prediction_speed="Fast",
        interpretability="Low",
        variants=["xgboost_binary_classification", "xgboost_multiclass_classification", "xgboost_regression"]
    ),

    "lightgbm": ModelInfo(
        id="lightgbm",
        display_name="LightGBM",
        category=ModelCategory.GRADIENT_BOOSTING,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="💨",
        short_description="Ultra-fast gradient boosting (3 variants)",
        long_description=(
            "LightGBM uses histogram-based algorithms and leaf-wise tree growth "
            "for 10-20x faster training than XGBoost on large datasets. "
            "Lower memory usage with similar accuracy."
        ),
        parameters=[
            ModelParameter(
                name="n_estimators",
                display_name="Number of Trees",
                description="Number of boosting rounds",
                default=100,
                range="50 to 500",
                example="100"
            ),
            ModelParameter(
                name="num_leaves",
                display_name="Number of Leaves",
                description="Maximum leaves per tree (leaf-wise growth)",
                default=31,
                range="20 to 150",
                example="31"
            ),
            ModelParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Step size shrinkage",
                default=0.1,
                range="0.01 to 0.3",
                example="0.1"
            )
        ],
        use_cases=[
            "Large datasets (>100K rows)",
            "Speed-critical applications",
            "Memory-constrained environments",
            "Real-time training"
        ],
        strengths=[
            "10-20x faster training",
            "50% less memory",
            "Often better accuracy",
            "GPU support"
        ],
        limitations=[
            "Can overfit on small data",
            "Leaf-wise vs depth-wise growth",
            "Different parameter names"
        ],
        requires_tuning=True,
        training_speed="Fast",
        prediction_speed="Fast",
        interpretability="Low",
        variants=["lightgbm_binary_classification", "lightgbm_multiclass_classification", "lightgbm_regression"]
    ),

    "catboost": ModelInfo(
        id="catboost",
        display_name="CatBoost",
        category=ModelCategory.GRADIENT_BOOSTING,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🐱",
        short_description="Gradient boosting with native categorical support (3 variants)",
        long_description=(
            "CatBoost handles categorical features natively without encoding. "
            "Robust to overfitting with ordered boosting and symmetric trees. "
            "Often requires less hyperparameter tuning than XGBoost/LightGBM."
        ),
        parameters=[
            ModelParameter(
                name="iterations",
                display_name="Number of Trees",
                description="Number of boosting iterations",
                default=100,
                range="50 to 500",
                example="100"
            ),
            ModelParameter(
                name="depth",
                display_name="Tree Depth",
                description="Maximum tree depth",
                default=6,
                range="4 to 10",
                example="6"
            ),
            ModelParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Step size shrinkage",
                default=0.1,
                range="0.01 to 0.3",
                example="0.1"
            )
        ],
        use_cases=[
            "Categorical features",
            "Production deployment",
            "Less tuning needed",
            "Robust predictions"
        ],
        strengths=[
            "Native categorical handling",
            "Less overfitting",
            "Built-in cross-validation",
            "Good default parameters"
        ],
        limitations=[
            "Slower than LightGBM",
            "Larger model size",
            "Different API"
        ],
        requires_tuning=False,
        training_speed="Medium",
        prediction_speed="Fast",
        interpretability="Low",
        variants=["catboost_binary_classification", "catboost_multiclass_classification", "catboost_regression"]
    ),

    # ========================================
    # NEURAL NETWORKS (2)
    # ========================================
    "keras_binary": ModelInfo(
        id="keras_binary_classification",
        display_name="Neural Network (Binary)",
        category=ModelCategory.NEURAL_NETWORKS,
        task_type=TaskType.BINARY_CLASSIFICATION,
        icon="🧠",
        short_description="Deep learning for complex binary classification",
        long_description=(
            "Keras/TensorFlow neural network for binary classification. "
            "Highly flexible architecture with multiple hidden layers. "
            "Best for complex non-linear patterns with large datasets."
        ),
        parameters=[
            ModelParameter(
                name="hidden_layers",
                display_name="Hidden Layers",
                description="Number and size of hidden layers",
                default="[64, 32]",
                range="[16] to [128, 64, 32]",
                example="[64, 32]"
            ),
            ModelParameter(
                name="epochs",
                display_name="Training Epochs",
                description="Number of training iterations",
                default=100,
                range="50 to 500",
                example="100"
            ),
            ModelParameter(
                name="learning_rate",
                display_name="Learning Rate",
                description="Optimizer learning rate",
                default=0.001,
                range="0.0001 to 0.01",
                example="0.001"
            )
        ],
        use_cases=[
            "Complex non-linear patterns",
            "Large datasets (>10K rows)",
            "Image/text embeddings",
            "Deep learning needed"
        ],
        strengths=[
            "Captures complex patterns",
            "Flexible architecture",
            "Transfer learning",
            "GPU acceleration"
        ],
        limitations=[
            "Requires large data",
            "Slow training",
            "Not interpretable",
            "Many hyperparameters"
        ],
        requires_tuning=True,
        training_speed="Slow",
        prediction_speed="Fast",
        interpretability="Low",
        variants=None
    ),

    "keras_multiclass": ModelInfo(
        id="keras_multiclass_classification",
        display_name="Neural Network (Multiclass)",
        category=ModelCategory.NEURAL_NETWORKS,
        task_type=TaskType.MULTICLASS_CLASSIFICATION,
        icon="🧠",
        short_description="Deep learning for multi-class classification",
        long_description=(
            "Keras/TensorFlow neural network for multiclass classification. "
            "Uses softmax output layer for probability distribution across classes. "
            "Handles complex multi-class problems."
        ),
        parameters=[
            ModelParameter(
                name="hidden_layers",
                display_name="Hidden Layers",
                description="Number and size of hidden layers",
                default="[64, 32]",
                range="[16] to [128, 64, 32]",
                example="[64, 32]"
            ),
            ModelParameter(
                name="epochs",
                display_name="Training Epochs",
                description="Number of training iterations",
                default=100,
                range="50 to 500",
                example="100"
            )
        ],
        use_cases=[
            "Multi-class problems",
            "Complex patterns",
            "Image classification",
            "Text categorization"
        ],
        strengths=[
            "Multi-class support",
            "Complex patterns",
            "Probability outputs",
            "Flexible"
        ],
        limitations=[
            "Requires large data",
            "Slow training",
            "Many hyperparameters",
            "GPU recommended"
        ],
        requires_tuning=True,
        training_speed="Slow",
        prediction_speed="Fast",
        interpretability="Low",
        variants=None
    )
}


def get_all_models() -> List[ModelInfo]:
    """Get all models from catalog."""
    return list(MODEL_CATALOG.values())


def get_models_by_category(category: ModelCategory) -> List[ModelInfo]:
    """Get models filtered by category."""
    return [m for m in MODEL_CATALOG.values() if m.category == category]


def get_models_by_task(task_type: TaskType) -> List[ModelInfo]:
    """Get models filtered by task type."""
    return [m for m in MODEL_CATALOG.values() if m.task_type == task_type]


def get_model_by_id(model_id: str) -> Optional[ModelInfo]:
    """Get model by technical ID."""
    return MODEL_CATALOG.get(model_id)


def search_models(query: str) -> List[ModelInfo]:
    """Search models by name or description."""
    query_lower = query.lower()
    return [
        m for m in MODEL_CATALOG.values()
        if query_lower in m.display_name.lower() or query_lower in m.short_description.lower()
    ]
```

### 3.2 Grouping Strategy

**Primary Organization: By Category**
- Linear Models (5)
- Tree-Based (1)
- Ensemble (1)
- Gradient Boosting (4: sklearn, XGBoost, LightGBM, CatBoost)
- Neural Networks (2)
- Bayesian (1)

**Alternative View: By Task Type**
- Regression (5 + variants)
- Binary Classification (6 + variants)
- Multiclass Classification (variants)

---

## 4. Implementation Details

### 4.1 State Management Updates

**File:** `src/core/state_manager.py` (MODIFY - add 2 states)

```python
class ModelsBrowserState(Enum):
    """States for models browser workflow."""
    VIEWING_MODEL_LIST = "viewing_model_list"      # Paginated list
    VIEWING_MODEL_DETAILS = "viewing_model_details"  # Single model details


# Add to WorkflowType enum
class WorkflowType(Enum):
    # ...existing...
    MODELS_BROWSER = "models_browser"  # NEW


# Add transitions
MODELS_BROWSER_TRANSITIONS: Dict[Optional[str], Set[str]] = {
    None: {ModelsBrowserState.VIEWING_MODEL_LIST.value},
    ModelsBrowserState.VIEWING_MODEL_LIST.value: {
        ModelsBrowserState.VIEWING_MODEL_DETAILS.value
    },
    ModelsBrowserState.VIEWING_MODEL_DETAILS.value: {
        ModelsBrowserState.VIEWING_MODEL_LIST.value  # Back button
    }
}

WORKFLOW_TRANSITIONS: Dict[WorkflowType, Dict[Optional[str], Set[str]]] = {
    # ...existing...
    WorkflowType.MODELS_BROWSER: MODELS_BROWSER_TRANSITIONS,
}
```

### 4.2 Handler Implementation

**File:** `src/bot/ml_handlers/models_browser_handler.py` (NEW - 250 lines)

```python
"""Telegram bot handlers for /models command - interactive model browser."""

import logging
from typing import Optional, List
from math import ceil

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from src.core.state_manager import StateManager, ModelsBrowserState, WorkflowType
from src.engines.model_catalog import (
    MODEL_CATALOG, ModelInfo, get_all_models, get_model_by_id
)
from src.bot.messages.models_messages import ModelsMessages

logger = logging.getLogger(__name__)

# Pagination settings
MODELS_PER_PAGE = 4  # Telegram best practice: 4-5 buttons per screen


class ModelsBrowserHandler:
    """Handler for /models command - browse ML model catalog."""

    def __init__(self, state_manager: StateManager):
        """Initialize handler with state manager."""
        self.state_manager = state_manager
        self.logger = logger

    async def handle_models_command(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle /models command - show paginated model list.

        Entry point for models browser workflow.
        """
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object in handle_models_command: {e}")
            if update and update.effective_message:
                await update.effective_message.reply_text(
                    "❌ **Invalid Request**\n\nPlease try /models again.",
                    parse_mode="Markdown"
                )
            return

        # Get or create session
        session = await self.state_manager.get_or_create_session(
            user_id=user_id,
            conversation_id=f"chat_{chat_id}"
        )

        # Initialize workflow at VIEWING_MODEL_LIST state
        session.workflow_type = WorkflowType.MODELS_BROWSER
        session.current_state = ModelsBrowserState.VIEWING_MODEL_LIST.value
        await self.state_manager.update_session(session)

        # Show first page of models
        await self._show_model_list(update, context, page=0)

    async def _show_model_list(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        page: int = 0
    ) -> None:
        """
        Show paginated list of models.

        Args:
            update: Telegram update
            context: Bot context
            page: Page number (0-indexed)
        """
        # Get all models
        all_models = get_all_models()
        total_models = len(all_models)
        total_pages = ceil(total_models / MODELS_PER_PAGE)

        # Validate page number
        page = max(0, min(page, total_pages - 1))

        # Get models for current page
        start_idx = page * MODELS_PER_PAGE
        end_idx = start_idx + MODELS_PER_PAGE
        page_models = all_models[start_idx:end_idx]

        # Build inline keyboard
        keyboard = []

        # Model buttons (one per row)
        for model in page_models:
            # Display with icon and name
            # Callback format: "model:id:page" (for back navigation)
            button_text = f"{model.icon} {model.display_name}"
            callback_data = f"model:{model.id}:{page}"
            keyboard.append([InlineKeyboardButton(button_text, callback_data=callback_data)])

        # Pagination + Cancel row
        nav_row = []
        if page > 0:
            nav_row.append(InlineKeyboardButton("← Prev", callback_data=f"page:{page-1}"))
        nav_row.append(InlineKeyboardButton("✖️ Cancel", callback_data="cancel_models"))
        if page < total_pages - 1:
            nav_row.append(InlineKeyboardButton("Next →", callback_data=f"page:{page+1}"))
        keyboard.append(nav_row)

        reply_markup = InlineKeyboardMarkup(keyboard)

        # Build message
        message_text = ModelsMessages.models_list_message(
            page=page + 1,
            total_pages=total_pages,
            total_models=total_models
        )

        # Send or edit message
        if update.message:
            # New command - send message
            await update.message.reply_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )
        else:
            # Callback query - edit existing message
            query = update.callback_query
            await query.answer()
            await query.edit_message_text(
                message_text,
                reply_markup=reply_markup,
                parse_mode="Markdown"
            )

    async def handle_model_selection(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle model selection - show model details.

        Callback format: "model:id:page"
        """
        query = update.callback_query
        await query.answer()

        # Parse callback data
        try:
            _, model_id, page_str = query.data.split(":")
            page = int(page_str)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid callback data in handle_model_selection: {e}")
            await query.edit_message_text(
                "❌ **Error**\n\nInvalid model selection. Please try /models again.",
                parse_mode="Markdown"
            )
            return

        # Get user session
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Save state snapshot for back navigation
        session.save_state_snapshot()

        # Transition to VIEWING_MODEL_DETAILS
        session.current_state = ModelsBrowserState.VIEWING_MODEL_DETAILS.value
        session.selections['current_page'] = page  # Store page for back navigation
        await self.state_manager.update_session(session)

        # Get model info
        model = get_model_by_id(model_id)
        if not model:
            await query.edit_message_text(
                f"❌ **Model Not Found**\n\nModel `{model_id}` not found in catalog.",
                parse_mode="Markdown"
            )
            return

        # Build message
        message_text = ModelsMessages.model_details_message(model)

        # Build keyboard with Back button
        keyboard = [[InlineKeyboardButton("← Back to Models", callback_data=f"back_to_list:{page}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Edit message with details
        await query.edit_message_text(
            message_text,
            reply_markup=reply_markup,
            parse_mode="Markdown"
        )

    async def handle_pagination(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle pagination callbacks.

        Callback format: "page:number"
        """
        query = update.callback_query
        await query.answer()

        # Parse page number
        try:
            _, page_str = query.data.split(":")
            page = int(page_str)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid pagination callback: {e}")
            return

        # Show requested page
        await self._show_model_list(update, context, page=page)

    async def handle_back_to_list(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle back button - return to model list.

        Callback format: "back_to_list:page"
        """
        query = update.callback_query
        await query.answer()

        # Parse page number
        try:
            _, page_str = query.data.split(":")
            page = int(page_str)
        except (ValueError, IndexError) as e:
            logger.error(f"Invalid back callback: {e}")
            page = 0

        # Get user session
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Restore previous state
        session.restore_previous_state()
        await self.state_manager.update_session(session)

        # Show model list
        await self._show_model_list(update, context, page=page)

    async def handle_cancel(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """
        Handle cancel button - exit models browser.

        Callback format: "cancel_models"
        """
        query = update.callback_query
        await query.answer()

        # Get user session
        try:
            user_id = update.effective_user.id
            chat_id = update.effective_chat.id
        except AttributeError as e:
            logger.error(f"Malformed update object: {e}")
            return

        session = await self.state_manager.get_session(user_id, f"chat_{chat_id}")

        # Cancel workflow
        await self.state_manager.cancel_workflow(session)

        # Edit message
        await query.edit_message_text(
            "✅ **Models Browser Closed**\n\nUse /models to browse models again.",
            parse_mode="Markdown"
        )
```

### 4.3 Message Templates

**File:** `src/bot/messages/models_messages.py` (NEW - 180 lines)

```python
"""Message templates for /models command."""

from typing import List
from src.engines.model_catalog import ModelInfo


class ModelsMessages:
    """Message formatting for models browser workflow."""

    @staticmethod
    def models_list_message(
        page: int,
        total_pages: int,
        total_models: int
    ) -> str:
        """
        Format models list message.

        Args:
            page: Current page number (1-indexed)
            total_pages: Total number of pages
            total_models: Total number of models

        Returns:
            Formatted message
        """
        return (
            f"📚 **ML Model Catalog**\n\n"
            f"Browse {total_models} available models for training.\n"
            f"Click a model to see details, parameters, and use cases.\n\n"
            f"**Page {page}/{total_pages}**\n\n"
            f"💡 **Tip:** Models are organized by type and task."
        )

    @staticmethod
    def model_details_message(model: ModelInfo) -> str:
        """
        Format model details message.

        Args:
            model: Model information

        Returns:
            Formatted message with full model details
        """
        # Header
        msg = f"{model.icon} **{model.display_name}**\n\n"

        # Short description
        msg += f"_{model.short_description}_\n\n"

        # Category and task type
        msg += f"**Category:** {model.category.value.replace('_', ' ').title()}\n"
        msg += f"**Task Type:** {model.task_type.value.replace('_', ' ').title()}\n\n"

        # Variants (if applicable)
        if model.variants:
            msg += "**Variants:**\n"
            for variant in model.variants:
                variant_display = variant.replace('_', ' ').replace(model.id, '').strip()
                msg += f"  • {variant_display.title()}\n"
            msg += "\n"

        # Long description
        msg += f"**Description:**\n{model.long_description}\n\n"

        # Performance characteristics
        msg += "**Performance:**\n"
        msg += f"  • Training Speed: {model.training_speed}\n"
        msg += f"  • Prediction Speed: {model.prediction_speed}\n"
        msg += f"  • Interpretability: {model.interpretability}\n"
        if model.requires_tuning:
            msg += f"  • Requires Tuning: ✅ Yes\n"
        else:
            msg += f"  • Requires Tuning: ❌ No\n"
        msg += "\n"

        # Key parameters
        if model.parameters:
            msg += "**Key Parameters:**\n"
            for param in model.parameters[:3]:  # Show top 3
                msg += f"  • **{param.display_name}** (`{param.name}`)\n"
                msg += f"    {param.description}\n"
                msg += f"    Default: `{param.default}`, Range: `{param.range}`\n"
            if len(model.parameters) > 3:
                msg += f"  ... and {len(model.parameters) - 3} more parameters\n"
            msg += "\n"

        # Use cases
        if model.use_cases:
            msg += "**Best For:**\n"
            for use_case in model.use_cases[:4]:  # Show top 4
                msg += f"  • {use_case}\n"
            msg += "\n"

        # Strengths
        if model.strengths:
            msg += "**Strengths:**\n"
            for strength in model.strengths[:3]:  # Show top 3
                msg += f"  ✅ {strength}\n"
            msg += "\n"

        # Limitations
        if model.limitations:
            msg += "**Limitations:**\n"
            for limitation in model.limitations[:3]:  # Show top 3
                msg += f"  ⚠️ {limitation}\n"
            msg += "\n"

        # Footer
        msg += "💡 **Tip:** Use `/train` to start training with this model."

        return msg
```

### 4.4 Command Registration

**File:** `src/bot/telegram_bot.py` (MODIFY - add command handler)

```python
# Add import
from src.bot.ml_handlers.models_browser_handler import ModelsBrowserHandler

# In setup_handlers() or equivalent:
models_handler = ModelsBrowserHandler(state_manager)

# Register command
application.add_handler(CommandHandler("models", models_handler.handle_models_command))

# Register callback query handlers
application.add_handler(CallbackQueryHandler(
    models_handler.handle_model_selection,
    pattern="^model:"
))
application.add_handler(CallbackQueryHandler(
    models_handler.handle_pagination,
    pattern="^page:"
))
application.add_handler(CallbackQueryHandler(
    models_handler.handle_back_to_list,
    pattern="^back_to_list:"
))
application.add_handler(CallbackQueryHandler(
    models_handler.handle_cancel,
    pattern="^cancel_models$"
))
```

---

## 5. Testing Strategy

### 5.1 Unit Tests

**File:** `tests/unit/test_models_browser.py` (NEW - 150 lines)

```python
"""Unit tests for models browser handler."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from src.bot.ml_handlers.models_browser_handler import ModelsBrowserHandler
from src.core.state_manager import StateManager, ModelsBrowserState
from src.engines.model_catalog import MODEL_CATALOG, get_all_models


class TestModelsBrowserHandler:
    """Test models browser handler functionality."""

    @pytest.fixture
    def state_manager(self):
        """Mock state manager."""
        return AsyncMock(spec=StateManager)

    @pytest.fixture
    def handler(self, state_manager):
        """Create handler instance."""
        return ModelsBrowserHandler(state_manager)

    @pytest.mark.asyncio
    async def test_handle_models_command(self, handler, state_manager):
        """Test /models command initializes workflow."""
        update = Mock()
        update.effective_user.id = 12345
        update.effective_chat.id = 67890
        update.message = Mock()

        context = Mock()

        # Mock session
        session = Mock()
        state_manager.get_or_create_session.return_value = session

        await handler.handle_models_command(update, context)

        # Verify workflow initialization
        state_manager.get_or_create_session.assert_called_once()
        assert session.workflow_type.value == "models_browser"
        assert session.current_state == ModelsBrowserState.VIEWING_MODEL_LIST.value

    @pytest.mark.asyncio
    async def test_pagination_logic(self, handler):
        """Test pagination correctly splits models."""
        all_models = get_all_models()
        models_per_page = 4
        total_pages = (len(all_models) + models_per_page - 1) // models_per_page

        # Test page boundaries
        assert total_pages >= 3  # Should have at least 3 pages
        page_0_models = all_models[0:4]
        assert len(page_0_models) == 4

    @pytest.mark.asyncio
    async def test_model_details_display(self, handler):
        """Test model details message formatting."""
        from src.bot.messages.models_messages import ModelsMessages

        # Get first model
        model = get_all_models()[0]

        # Generate message
        message = ModelsMessages.model_details_message(model)

        # Verify key sections
        assert model.display_name in message
        assert model.short_description in message
        assert "Parameters:" in message or "Best For:" in message
        assert "Tip:" in message

    def test_callback_data_length(self):
        """Verify callback data stays under Telegram's 64-byte limit."""
        # Test longest model ID
        longest_id = max(MODEL_CATALOG.keys(), key=len)
        callback = f"model:{longest_id}:99"  # page 99
        assert len(callback.encode('utf-8')) < 64, f"Callback data too long: {len(callback)} bytes"

    @pytest.mark.asyncio
    async def test_back_button_navigation(self, handler, state_manager):
        """Test back button restores previous state."""
        update = Mock()
        update.callback_query = Mock()
        update.callback_query.data = "back_to_list:2"
        update.effective_user.id = 12345
        update.effective_chat.id = 67890

        context = Mock()

        # Mock session
        session = Mock()
        session.restore_previous_state = Mock(return_value=True)
        state_manager.get_session.return_value = session

        await handler.handle_back_to_list(update, context)

        # Verify state restoration
        session.restore_previous_state.assert_called_once()
```

### 5.2 Integration Tests

**File:** `tests/integration/test_models_workflow.py` (NEW - 100 lines)

```python
"""Integration tests for /models workflow."""

import pytest
from unittest.mock import AsyncMock, Mock

from src.bot.ml_handlers.models_browser_handler import ModelsBrowserHandler
from src.core.state_manager import StateManager, StateManagerConfig


class TestModelsWorkflowIntegration:
    """Test complete /models workflow end-to-end."""

    @pytest.fixture
    def state_manager(self):
        """Create real state manager for integration testing."""
        config = StateManagerConfig(session_timeout_minutes=30)
        return StateManager(config)

    @pytest.fixture
    def handler(self, state_manager):
        """Create handler with real state manager."""
        return ModelsBrowserHandler(state_manager)

    @pytest.mark.asyncio
    async def test_full_workflow(self, handler, state_manager):
        """Test complete workflow: list → details → back → cancel."""
        # Step 1: User enters /models
        update_cmd = Mock()
        update_cmd.effective_user.id = 12345
        update_cmd.effective_chat.id = 67890
        update_cmd.message = AsyncMock()
        context = Mock()

        await handler.handle_models_command(update_cmd, context)

        # Verify: Message sent with model list
        update_cmd.message.reply_text.assert_called_once()
        message_text = update_cmd.message.reply_text.call_args[0][0]
        assert "ML Model Catalog" in message_text

        # Step 2: User clicks model
        update_select = Mock()
        update_select.callback_query = AsyncMock()
        update_select.callback_query.data = "model:linear:0"
        update_select.effective_user.id = 12345
        update_select.effective_chat.id = 67890

        await handler.handle_model_selection(update_select, context)

        # Verify: Details shown
        update_select.callback_query.edit_message_text.assert_called_once()
        details_text = update_select.callback_query.edit_message_text.call_args[0][0]
        assert "Linear Regression" in details_text
        assert "Parameters:" in details_text

        # Step 3: User clicks back
        update_back = Mock()
        update_back.callback_query = AsyncMock()
        update_back.callback_query.data = "back_to_list:0"
        update_back.effective_user.id = 12345
        update_back.effective_chat.id = 67890

        await handler.handle_back_to_list(update_back, context)

        # Verify: Back to list
        update_back.callback_query.edit_message_text.assert_called_once()
        list_text = update_back.callback_query.edit_message_text.call_args[0][0]
        assert "ML Model Catalog" in list_text

        # Step 4: User clicks cancel
        update_cancel = Mock()
        update_cancel.callback_query = AsyncMock()
        update_cancel.callback_query.data = "cancel_models"
        update_cancel.effective_user.id = 12345
        update_cancel.effective_chat.id = 67890

        await handler.handle_cancel(update_cancel, context)

        # Verify: Workflow cancelled
        update_cancel.callback_query.edit_message_text.assert_called_once()
        cancel_text = update_cancel.callback_query.edit_message_text.call_args[0][0]
        assert "Models Browser Closed" in cancel_text

    @pytest.mark.asyncio
    async def test_pagination_workflow(self, handler):
        """Test pagination through multiple pages."""
        # Test Next button
        update_next = Mock()
        update_next.callback_query = AsyncMock()
        update_next.callback_query.data = "page:1"
        context = Mock()

        await handler.handle_pagination(update_next, context)

        # Verify page 2 shown
        update_next.callback_query.edit_message_text.assert_called_once()
        page2_text = update_next.callback_query.edit_message_text.call_args[0][0]
        assert "Page 2/" in page2_text
```

---

## 6. Technical Considerations

### 6.1 Telegram API Constraints

**Callback Data Limit: 64 bytes**
- Current format: `"model:xgboost_binary_classification:99"` = ~37 bytes ✅
- Solution: Use short IDs, store page in session if needed

**Message Length Limit: 4096 characters**
- Model details message: ~800-1200 characters ✅
- Solution: Truncate long descriptions if needed

**Inline Keyboard Limits: 8 buttons per row, 100 total**
- Current design: 4 models + 1 navigation row = 5 rows ✅
- Solution: Pagination keeps under limit

### 6.2 State Persistence

**Back Button Implementation:**
```python
# Before showing details:
session.save_state_snapshot()  # Saves current page

# On back button:
session.restore_previous_state()  # Restores to list view with page
```

**Session Data Storage:**
- `selections['current_page']`: Current page number
- `current_state`: VIEWING_MODEL_LIST or VIEWING_MODEL_DETAILS
- State history stack handles navigation

### 6.3 Error Handling

**Scenarios:**
1. **Invalid model ID**: Show error, return to list
2. **Invalid page number**: Clamp to valid range (0 to total_pages-1)
3. **Malformed callback data**: Log error, ask user to restart
4. **Session expired**: Graceful error, suggest /models again

### 6.4 Performance

**Model Catalog Loading:**
- Catalog is static Python dict → instant access
- No database queries needed
- Memory footprint: ~50KB for full catalog

**Message Generation:**
- Template-based formatting → <1ms per message
- No async operations needed for formatting

---

## 7. Integration Points

### 7.1 Connection to /train Workflow

**User Journey:**
```
/models → Browse → Find "XGBoost" → See details → /train → Select XGBoost
```

**Implementation:**
- Models browser is read-only (discovery tool)
- No direct transition to /train (keeps workflows separate)
- User manually uses /train after browsing

### 7.2 Model Catalog Sync

**Ensuring Catalog Accuracy:**
1. Model catalog (model_catalog.py) is source of truth
2. Trainers expose `SUPPORTED_MODELS` class attribute
3. Unit test verifies catalog matches trainer registrations

**Sync Test:**
```python
def test_catalog_matches_trainers():
    """Verify all models in catalog exist in trainers."""
    from src.engines.ml_engine import MLEngine

    ml_engine = MLEngine(MLEngineConfig.get_default())

    # Get all supported models from trainers
    all_trainer_models = set()
    for trainer in ml_engine.trainers.values():
        all_trainer_models.update(trainer.SUPPORTED_MODELS)

    # Get all models from catalog (excluding variants)
    catalog_models = {m.id for m in get_all_models() if not m.variants}

    # Verify match
    assert catalog_models.issubset(all_trainer_models), "Catalog contains unsupported models"
```

---

## 8. Rollout Strategy

### Phase 1: Core Implementation (6 hours)
1. ✅ Create model_catalog.py with all 13+ models
2. ✅ Implement ModelsBrowserHandler
3. ✅ Create ModelsMessages
4. ✅ Update state_manager.py

### Phase 2: Testing (2 hours)
5. ✅ Write 15 unit tests
6. ✅ Write 5 integration tests
7. ✅ Verify catalog sync
8. ✅ Test all edge cases

### Phase 3: Documentation (1 hour)
9. ✅ Save this plan to dev/implemented/models-menu.md
10. ✅ Update README.md
11. ✅ Add /models to help message

### Phase 4: Deployment (1 hour)
12. ✅ Test in development environment
13. ✅ Verify bot restart works
14. ✅ Monitor logs for errors
15. ✅ Deploy to production

---

## 9. Success Criteria

✅ **Functionality:**
- /models command shows paginated model list
- Clicking model shows full details
- Back button returns to correct page
- Cancel button closes browser
- All 13+ models accessible

✅ **Quality:**
- 20+ tests passing (15 unit + 5 integration)
- No Telegram API limit violations
- State persistence works correctly
- Error handling covers edge cases

✅ **User Experience:**
- Clear model descriptions
- Intuitive navigation
- Fast response times (<500ms)
- Mobile-friendly button layout

✅ **Code Quality:**
- Type hints on all functions
- Docstrings on all public methods
- Follows existing code patterns
- No pylint/mypy warnings

---

## 10. Future Enhancements

**Phase 2 Features (Future):**
1. **Search Functionality**: `/models search gradient` filters models
2. **Comparison View**: Compare 2-3 models side-by-side
3. **Favorites**: Save preferred models for quick access
4. **Recommendations**: "Best model for your data" based on dataset stats
5. **Direct Training**: "Train with this model" button → /train pre-filled

**Phase 3 Features (Future):**
6. **Interactive Tutorials**: Step-by-step guides per model
7. **Performance Benchmarks**: Real accuracy scores on public datasets
8. **Model Playground**: Test model on sample data before training

---

## 11. Summary

### Deliverables
- ✅ model_catalog.py (400 lines) - Complete model metadata
- ✅ models_browser_handler.py (250 lines) - Telegram handler logic
- ✅ models_messages.py (180 lines) - Message formatting
- ✅ state_manager.py updates (15 lines) - New states
- ✅ test_models_browser.py (150 lines) - Unit tests
- ✅ test_models_workflow.py (100 lines) - Integration tests

### Total Effort
- **Implementation**: 6 hours
- **Testing**: 2 hours
- **Documentation**: 1 hour
- **Deployment**: 1 hour
- **TOTAL**: ~10 hours

### Risk Assessment
- **Low Risk**: Well-defined requirements, existing patterns to follow
- **Medium Complexity**: Pagination logic, state management
- **High Value**: Improves model discoverability for users

---

**END OF IMPLEMENTATION PLAN**
