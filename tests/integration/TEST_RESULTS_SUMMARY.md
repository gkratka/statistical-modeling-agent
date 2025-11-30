# Model Type Comparison Tests - Initial Results

**Test File**: `tests/integration/test_all_model_types_cloud.py`  
**Created**: 2025-11-08  
**Task**: 6.7 - Model Type Comparison Tests

## Test Summary

**Total Tests**: 31  
**Passed**: 12 (38.7%)  
**Failed**: 19 (61.3%)  

### Test Categories

1. **Model Parity Tests (13 tests)**:
   - Regression models: 5/5 PASSED ✅
   - Classification models: 6/6 PASSED ✅
   - Neural network models: 0/2 FAILED ❌

2. **Artifact Verification Tests (13 tests)**:
   - All 13 tests: FAILED ❌ (due to test isolation issues)

3. **Preprocessing Consistency Tests (3 tests)**:
   - Missing value handling: 1/1 PASSED ✅
   - StandardScaler: 0/1 FAILED ❌
   - MinMaxScaler: 0/1 FAILED ❌

4. **Hyperparameter Handling Tests (2 tests)**:
   - Both FAILED ❌ (file not found issues)

## Passing Tests (12)

### Regression Model Parity (5)
- test_regression_model_parity[linear] ✅
- test_regression_model_parity[ridge] ✅
- test_regression_model_parity[lasso] ✅
- test_regression_model_parity[elasticnet] ✅
- test_regression_model_parity[polynomial] ✅

### Classification Model Parity (6)
- test_classification_model_parity[logistic] ✅
- test_classification_model_parity[decision_tree] ✅
- test_classification_model_parity[random_forest] ✅
- test_classification_model_parity[gradient_boosting] ✅
- test_classification_model_parity[svm] ✅
- test_classification_model_parity[naive_bayes] ✅

### Preprocessing (1)
- test_missing_value_handling_consistency ✅

## Failing Tests (19)

### Issues Identified

1. **Neural Network Task Type Error (2 tests)**:
   - mlp_regression/mlp_classification require `task_type='neural_network'`
   - Currently using `task_type='regression'/'classification'`
   - Error: `ValidationError: Model type 'mlp_regression' not supported for task 'regression'`

2. **Artifact File Not Found (15 tests)**:
   - All artifact verification tests failing
   - Preprocessor tests failing
   - Hyperparameter tests failing  
   - Root cause: Test isolation - different user_ids in same tmp_path
   - Models saved with user_id but tests look in different locations

3. **Missing Model Files**:
   - `model.pkl` not found in expected directories
   - `preprocessor.pkl` not found
   - `metadata.json` not found

## Root Cause Analysis

### Issue 1: Neural Network Task Type
The ML Engine requires `task_type='neural_network'` for MLP models, not 'regression'/'classification'.

**Fix**: Update neural network parity tests to use correct task_type.

### Issue 2: Test Isolation
Each test class is creating models with overlapping user_ids (40001-60002) but using separate tmp_path fixtures. This creates conflicts where:
- Test 1 creates user_40001 in tmp_path_1
- Test 2 tries to read user_40001 but gets tmp_path_2

**Fix**: Use unique user_ids per test or share fixture across test classes.

### Issue 3: Artifact Path Construction
Tests are constructing paths like `/tmp/.../models/user_40001/model_id/` but models may be saved differently.

**Fix**: Verify actual model directory structure and adjust path construction.

## Recommendations

1. **Fix neural network task type**: Change `task_type` parameter to 'neural_network' for MLP models
2. **Fix test isolation**: Use unique user IDs (70001-90000) for artifact/preprocessing/hyperparameter tests
3. **Debug artifact paths**: Add logging to show where models are actually saved
4. **Consolidate fixtures**: Share ml_engine fixture across test classes to maintain consistent tmp_path

## What Works

The core functionality is solid:
- ✅ All 11 regression/classification model parity tests pass
- ✅ Models train successfully
- ✅ Predictions match between "local" and "cloud" (simulated) training
- ✅ Metrics are consistent
- ✅ Missing value handling is consistent

## Next Steps

1. Update neural network tests to use `task_type='neural_network'`
2. Fix user_id allocation to ensure test isolation
3. Debug and fix artifact path resolution
4. Re-run tests to validate fixes
5. Achieve 31/31 passing tests

## File Location

`/Users/gkratka/Documents/statistical-modeling-agent/tests/integration/test_all_model_types_cloud.py`

## Model Types Tested

**Regression (5)**: linear, ridge, lasso, elasticnet, polynomial  
**Classification (6)**: logistic, decision_tree, random_forest, gradient_boosting, svm, naive_bayes  
**Neural Networks (2)**: mlp_regression, mlp_classification

**Total**: 13 model types
