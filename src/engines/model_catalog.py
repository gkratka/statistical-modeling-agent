"""ML Model Catalog - Complete metadata for all available models.

This module provides a centralized registry of all machine learning models
available in the bot, including detailed information about parameters,
use cases, strengths, and limitations.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Any
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
    default: Optional[Any]
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
    "keras_binary_classification": ModelInfo(
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

    "keras_multiclass_classification": ModelInfo(
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
