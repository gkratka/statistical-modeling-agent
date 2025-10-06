# Test Data

This directory contains sample datasets for testing the ML Engine.

## Available Datasets

### 1. housing_regression.csv
**Purpose**: Test regression models

**Columns**:
- `sqft`: Square footage of the house
- `bedrooms`: Number of bedrooms
- `bathrooms`: Number of bathrooms
- `age`: Age of the house in years
- `price`: Sale price (target variable)

**Use Cases**:
- Linear regression
- Ridge/Lasso regression
- Polynomial regression
- Feature importance analysis

### 2. customer_classification.csv
**Purpose**: Test classification models

**Columns**:
- `age`: Customer age
- `income`: Annual income
- `purchase_frequency`: Number of purchases per month
- `churned`: Whether customer churned (0=No, 1=Yes) - target variable

**Use Cases**:
- Logistic regression
- Decision tree classification
- Random forest classification
- Binary classification metrics

## Usage Examples

### Regression Example
```python
import pandas as pd
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig

data = pd.read_csv('test_data/housing_regression.csv')
engine = MLEngine(MLEngineConfig.get_default())

result = engine.train_model(
    data=data,
    task_type="regression",
    model_type="linear",
    target_column="price",
    feature_columns=["sqft", "bedrooms", "bathrooms", "age"],
    user_id=12345
)
```

### Classification Example
```python
import pandas as pd
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig

data = pd.read_csv('test_data/customer_classification.csv')
engine = MLEngine(MLEngineConfig.get_default())

result = engine.train_model(
    data=data,
    task_type="classification",
    model_type="logistic",
    target_column="churned",
    feature_columns=["age", "income", "purchase_frequency"],
    user_id=12345
)
```
