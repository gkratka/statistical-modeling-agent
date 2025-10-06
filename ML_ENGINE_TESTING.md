# ML Engine Testing Guide

## 🚀 Quick Start

### Run Tests
```bash
# Quick smoke test (5 tests, ~5 seconds)
python3 scripts/test_ml_quick.py

# Full test suite (34 tests, ~30 seconds)
python3 scripts/test_ml_engine_manual.py

# Run specific phase
python3 scripts/test_ml_engine_manual.py --phase 1

# List all phases
python3 scripts/test_ml_engine_manual.py --list
```

## ✅ Current Status

**All Tests Passing**: 34/34 (100%)

- ✓ Regression models (5 types)
- ✓ Classification models (6 types)
- ✓ Preprocessing (missing values, scaling)
- ✓ Model persistence
- ✓ Predictions
- ✓ Error handling

## 📊 Test Coverage

### Phase 1: Basic Regression (7 tests)
Linear, Ridge, Lasso, ElasticNet, Polynomial

### Phase 2: Preprocessing (7 tests)
Missing: mean, median, drop
Scaling: standard, minmax, robust, none

### Phase 3: Classification (7 tests)
Logistic, Decision Tree, Random Forest, Gradient Boosting, SVM, Naive Bayes

### Phase 4: Model Lifecycle (4 tests)
List, filter, info, delete

### Phase 5: Predictions (2 tests)
Regression, classification with probabilities

### Phase 6: Error Handling (4 tests)
Empty data, insufficient samples, invalid types

### Phase 7: Advanced (3 tests)
Multiple models, custom test sizes, supported models

## 🐛 Issues Fixed

1. **Models not saving** - Added ModelManager.save_model()
2. **Prediction validation error** - Fixed parameter passing
3. **Missing values drop strategy** - Reordered preprocessing

## 📁 Files Created

- `scripts/test_ml_quick.py` - Quick smoke test
- `scripts/test_ml_engine_manual.py` - Comprehensive test suite
- `test_data/housing_regression.csv` - Test data
- `test_data/customer_classification.csv` - Test data
- `claudedocs/ml_engine_testing_summary.md` - Detailed report

## 💡 Usage Example

```python
from src.engines.ml_engine import MLEngine
from src.engines.ml_config import MLEngineConfig
import pandas as pd

# Setup
config = MLEngineConfig.get_default()
engine = MLEngine(config)
data = pd.DataFrame({'x': range(20), 'y': [i*2 for i in range(20)]})

# Train
result = engine.train_model(
    data=data,
    task_type="regression",
    model_type="linear",
    target_column="y",
    feature_columns=["x"],
    user_id=12345
)
print(f"Model ID: {result['model_id']}")
print(f"R²: {result['metrics']['r2']}")

# Predict
new_data = pd.DataFrame({'x': [5, 10, 15]})
predictions = engine.predict(
    user_id=12345,
    model_id=result['model_id'],
    data=new_data
)
print(f"Predictions: {predictions['predictions']}")

# Manage
models = engine.list_models(user_id=12345)
print(f"Total models: {len(models)}")
```

## 🔍 Debugging Tips

1. **Check model directory**: `ls -la models/user_{user_id}/`
2. **View metadata**: `cat models/user_{user_id}/{model_id}/metadata.json`
3. **Run specific phase**: `python3 scripts/test_ml_engine_manual.py --phase 2`
4. **Add print statements** in test scripts for detailed debugging

## 📈 Next Steps

1. Test neural network models (MLP)
2. Integrate with Telegram bot
3. Add cross-validation testing
4. Performance testing with large datasets
5. Script generation and sandboxed execution

---

For detailed information, see: `claudedocs/ml_engine_testing_summary.md`
