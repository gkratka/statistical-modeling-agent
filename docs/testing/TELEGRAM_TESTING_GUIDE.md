# Telegram Bot Manual Testing Guide

## 📋 Overview

This guide provides comprehensive manual testing procedures for the Statistical Modeling Agent Telegram bot. Use this to verify all ML Engine functionality is working correctly through the Telegram interface.

**Testing Environment**: Telegram mobile app or desktop client
**Prerequisites**: Bot must be running, .env configured with TELEGRAM_BOT_TOKEN

---

## 🎯 Testing Checklist

### Phase 1: Basic Functionality ✅
- [ ] Bot startup and /start command
- [ ] /help command
- [ ] File upload validation
- [ ] Data inspection queries

### Phase 2: Statistics ✅
- [ ] Descriptive statistics
- [ ] Correlation analysis

### Phase 3: ML Training ✅
- [ ] Regression models (5 tests)
- [ ] Classification models (6 tests)
- [ ] Neural networks (2 tests)

### Phase 4: ML Operations ✅
- [ ] Model predictions
- [ ] Model listing
- [ ] Model information
- [ ] Error handling

---

## 📂 Test Data Preparation

### Dataset 1: Housing Regression Data
**Purpose**: Test regression models
**File**: `housing_data.csv`

```csv
sqft,bedrooms,bathrooms,age,price
1200,2,1,10,250000
1800,3,2,5,350000
2400,4,3,2,450000
1000,2,1,15,200000
1500,3,2,8,280000
2000,3,2,6,380000
2200,4,3,3,420000
1100,2,1,12,230000
1600,3,2,7,300000
1900,3,2,4,360000
2100,4,3,5,400000
1300,2,2,9,260000
1700,3,2,6,320000
2300,4,3,3,440000
1400,3,2,8,270000
1850,3,2,5,340000
2050,4,3,4,390000
1250,2,1,11,240000
1550,3,2,7,290000
1950,3,2,5,370000
```

**Columns**:
- `sqft` (numeric) - Square footage
- `bedrooms` (numeric) - Number of bedrooms
- `bathrooms` (numeric) - Number of bathrooms
- `age` (numeric) - Age of property in years
- `price` (numeric) - **TARGET** - Price in dollars

---

### Dataset 2: Iris Classification Data
**Purpose**: Test classification models
**File**: `iris_data.csv`

```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
4.7,3.2,1.3,0.2,setosa
4.6,3.1,1.5,0.2,setosa
5.0,3.6,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
6.4,3.2,4.5,1.5,versicolor
6.9,3.1,4.9,1.5,versicolor
5.5,2.3,4.0,1.3,versicolor
6.5,2.8,4.6,1.5,versicolor
6.3,3.3,6.0,2.5,virginica
5.8,2.7,5.1,1.9,virginica
7.1,3.0,5.9,2.1,virginica
6.3,2.9,5.6,1.8,virginica
6.5,3.0,5.8,2.2,virginica
4.9,3.1,1.5,0.1,setosa
5.4,3.7,1.5,0.2,setosa
6.0,2.9,4.5,1.5,versicolor
5.7,2.8,4.5,1.3,versicolor
6.7,3.1,5.6,2.4,virginica
```

**Columns**:
- `sepal_length` (numeric)
- `sepal_width` (numeric)
- `petal_length` (numeric)
- `petal_width` (numeric)
- `species` (categorical) - **TARGET** - setosa/versicolor/virginica

---

### Dataset 3: Employee Analytics Data
**Purpose**: Test mixed scenarios
**File**: `employee_data.csv`

```csv
employee_id,age,salary,experience,performance_score,department,promoted
1,25,50000,2,3.5,Engineering,0
2,30,65000,5,4.2,Marketing,1
3,35,75000,8,4.5,Sales,1
4,28,55000,3,3.8,HR,0
5,32,70000,6,4.0,Engineering,1
6,27,52000,2,3.6,Marketing,0
7,40,85000,12,4.7,Sales,1
8,29,58000,4,3.9,HR,0
9,33,72000,7,4.3,Engineering,1
10,26,51000,2,3.4,Marketing,0
11,38,80000,10,4.6,Sales,1
12,31,68000,5,4.1,HR,1
13,24,48000,1,3.2,Engineering,0
14,36,78000,9,4.4,Marketing,1
15,34,74000,7,4.2,Sales,1
```

**Columns**:
- `employee_id` (numeric)
- `age` (numeric)
- `salary` (numeric) - Can use as regression target
- `experience` (numeric)
- `performance_score` (numeric)
- `department` (categorical)
- `promoted` (binary: 0/1) - **TARGET** for classification

---

## 🧪 Test Scenarios

### Phase 1: Basic Functionality

#### Test 1.1: Bot Startup
**Command**: `/start`

**Expected Response**:
```
🤖 Welcome to the Statistical Modeling Agent!
🔧 Version: DataLoader-v2.0-NUCLEAR-FIX
🔧 Instance: BIH-2025-01-27-NUCLEAR

I can help you with:
📊 Statistical analysis of your data
🧠 Machine learning model training
📈 Data predictions and insights

To get started:
1. Upload a CSV file with your data
2. Tell me what analysis you'd like
3. I'll process it and send you results!

Type /help for more information.
```

**Success Criteria**:
- ✅ Bot responds immediately
- ✅ Version information displayed
- ✅ Welcome message clear and formatted

---

#### Test 1.2: Help Command
**Command**: `/help`

**Expected Response**:
```
🆘 Statistical Modeling Agent Help

Commands:
/start - Start using the bot
/help - Show this help message

How to use:
1. Upload Data: Send a CSV file
2. Request Analysis: Tell me what you want:
   • Calculate mean and std for age column
   • Show correlation matrix
   • Train a model to predict income
3. Get Results: I'll analyze and respond

Supported Operations:
📊 Descriptive statistics
📈 Correlation analysis
🧠 Machine learning training
🔮 Model predictions

Example:
1. Upload: housing_data.csv
2. Message: Train a model to predict house prices
3. Get: Model training results and performance metrics

Need more help? Just ask me anything!
```

**Success Criteria**:
- ✅ Help text displays all commands
- ✅ Examples are clear
- ✅ Formatting is readable

---

#### Test 1.3: File Upload (Housing Data)
**Action**: Upload `housing_data.csv` as a file attachment

**Expected Response**:
```
✅ Data loaded successfully!

📊 File: housing_data.csv
📈 Size: 20 rows × 5 columns

Columns:
• sqft (numeric)
• bedrooms (numeric)
• bathrooms (numeric)
• age (numeric)
• price (numeric)

🎯 Ready for analysis! Try asking:
• "Calculate statistics for price"
• "Show correlation matrix"
• "Train a model to predict price"

📋 Sample data:
   sqft  bedrooms  bathrooms  age   price
0  1200         2          1   10  250000
1  1800         3          2    5  350000
2  2400         4          3    2  450000
```

**Success Criteria**:
- ✅ File uploads without errors
- ✅ Row and column counts are correct
- ✅ All columns detected with correct types
- ✅ Sample data preview shown
- ✅ Helpful suggestions provided

---

#### Test 1.4: Data Inspection Query
**Prerequisites**: Housing data uploaded
**Command**: `what columns are in my data?`

**Expected Response**:
```
📊 Your Data: housing_data.csv

Columns (5):
• sqft
• bedrooms
• bathrooms
• age
• price

Data Shape: 20 rows × 5 columns

Try asking:
• "Calculate statistics for sqft"
• "Show correlation matrix"
• "Train a model to predict price"

🔧 DataLoader v2.0 active - Ready for analysis!
```

**Success Criteria**:
- ✅ All columns listed
- ✅ Row/column counts correct
- ✅ Suggestions provided

---

### Phase 2: Statistics Testing

#### Test 2.1: Descriptive Statistics
**Prerequisites**: Housing data uploaded
**Command**: `calculate statistics for price`

**Expected Response** (format may vary):
```
✅ Analysis Complete

Operation: descriptive_stats
Execution Time: 0.5s

Results for column 'price':
• Count: 20
• Mean: 321,000
• Median: 330,000
• Std Dev: 72,456
• Min: 200,000
• Max: 450,000
• Q1: 262,500
• Q3: 392,500

📊 Distribution:
The price data shows moderate variability with a standard deviation
of $72,456 around a mean of $321,000.
```

**Success Criteria**:
- ✅ All statistics calculated correctly
- ✅ Values are reasonable for the data
- ✅ Execution time < 2 seconds
- ✅ Formatting is clear

**Alternative Commands to Try**:
- `show me descriptive stats for sqft`
- `give me statistics for all numeric columns`
- `calculate mean, median, and std for bedrooms`

---

#### Test 2.2: Correlation Analysis
**Prerequisites**: Housing data uploaded
**Command**: `show correlation between price and sqft`

**Expected Response**:
```
✅ Correlation Analysis Complete

Correlation Matrix:
           price     sqft
price     1.000    0.95
sqft      0.95     1.000

Key Findings:
• Strong positive correlation (0.95) between price and sqft
• Relationship is statistically significant
• As sqft increases, price tends to increase

Execution Time: 0.6s
```

**Success Criteria**:
- ✅ Correlation coefficient calculated
- ✅ Value is between -1 and 1
- ✅ Matrix is symmetric
- ✅ Interpretation provided

**Alternative Commands**:
- `show correlation matrix for all columns`
- `correlate price with bedrooms and bathrooms`

---

### Phase 3: ML Training Tests

#### Test 3.1: Linear Regression
**Prerequisites**: Housing data uploaded
**Command**: `train a linear regression model to predict price using sqft, bedrooms, and bathrooms`

**Expected Response**:
```
✅ Model Training Complete

Model ID: model_12345_linear
Task Type: regression
Model Type: linear

Training Metrics:
• R² Score: 0.82
• Mean Squared Error: 8,234,567
• Root Mean Squared Error: 2,870
• Mean Absolute Error: 2,145

Test Set Performance:
• Training Time: 1.2s
• Samples Used: 20 (16 train, 4 test)

Model Features:
• sqft
• bedrooms
• bathrooms

✅ Model saved successfully!
Use model ID 'model_12345_linear' for predictions.
```

**Success Criteria**:
- ✅ Model trains without errors
- ✅ R² score between 0 and 1
- ✅ MSE, RMSE, MAE reported
- ✅ Model ID provided
- ✅ Training time < 5 seconds
- ✅ Model saved confirmation

**What to Check**:
- R² > 0.7 indicates good fit
- RMSE should be reasonable relative to price range
- Model ID should be unique

---

#### Test 3.2: Ridge Regression
**Prerequisites**: Housing data uploaded
**Command**: `train a ridge model to predict price`

**Expected Response**: Similar to Test 3.1 but with:
```
Model Type: ridge
Training Metrics:
• R² Score: 0.80-0.85 (may differ slightly from linear)
• Regularization applied (alpha=1.0)
```

**Success Criteria**: Same as Test 3.1
**Note**: Ridge may have slightly lower R² than linear due to regularization

---

#### Test 3.3: Lasso Regression
**Prerequisites**: Housing data uploaded
**Command**: `use lasso regression to predict price from all features`

**Expected Response**: Similar format with:
```
Model Type: lasso
Training Metrics:
• R² Score: 0.78-0.84
• Feature selection applied
• Some coefficients may be zero
```

**Success Criteria**: Same as Test 3.1
**Note**: Lasso performs feature selection - some features may have zero coefficients

---

#### Test 3.4: Random Forest Regression
**Prerequisites**: Housing data uploaded
**Command**: `train random forest to predict price`

**Expected Response**:
```
✅ Model Training Complete

Model ID: model_12345_random_forest
Model Type: random_forest

Training Metrics:
• R² Score: 0.85-0.95 (typically higher than linear)
• RMSE: 1,800-2,500
• MAE: 1,500-2,000

Model Configuration:
• Number of Trees: 100 (default)
• Feature Importance Available: Yes

Top Features by Importance:
1. sqft: 0.65
2. bathrooms: 0.20
3. bedrooms: 0.15
```

**Success Criteria**:
- ✅ Model trains successfully
- ✅ R² typically higher than linear models
- ✅ Feature importance scores provided
- ✅ Scores sum to approximately 1.0

---

#### Test 3.5: Polynomial Regression
**Prerequisites**: Housing data uploaded
**Command**: `train polynomial regression for price prediction`

**Expected Response**:
```
Model Type: polynomial
Training Metrics:
• R² Score: 0.80-0.90
• Polynomial Degree: 2 (default)
• Can capture non-linear relationships
```

**Success Criteria**: Same as Test 3.1
**Note**: May overfit on small datasets

---

#### Test 3.6: Logistic Regression (Classification)
**Prerequisites**: Iris data uploaded
**Command**: `train logistic regression to predict species`

**Expected Response**:
```
✅ Model Training Complete

Model ID: model_12345_logistic
Task Type: classification
Model Type: logistic

Training Metrics:
• Accuracy: 0.90-1.00
• Precision: 0.92
• Recall: 0.91
• F1 Score: 0.91

Classes:
• setosa
• versicolor
• virginica

Confusion Matrix Available: Yes
Training Time: 0.8s
```

**Success Criteria**:
- ✅ Accuracy > 0.85
- ✅ All metrics between 0 and 1
- ✅ All 3 classes detected
- ✅ Multi-class classification working

---

#### Test 3.7: Decision Tree Classification
**Prerequisites**: Iris data uploaded
**Command**: `use decision tree to classify species`

**Expected Response**:
```
Model Type: decision_tree
Training Metrics:
• Accuracy: 0.90-1.00
• Tree depth reported
• Feature importance available
```

**Success Criteria**: Same as Test 3.6

---

#### Test 3.8: Random Forest Classification
**Prerequisites**: Iris data uploaded
**Command**: `train random forest classifier for species`

**Expected Response**:
```
Model Type: random_forest
Training Metrics:
• Accuracy: 0.95-1.00 (typically very high for Iris)
• Number of Trees: 100
• Feature Importance Available

Top Features:
1. petal_length: 0.45
2. petal_width: 0.42
3. sepal_length: 0.08
4. sepal_width: 0.05
```

**Success Criteria**:
- ✅ Accuracy > 0.90
- ✅ Feature importance ranked
- ✅ Petal features should rank highest for Iris

---

#### Test 3.9: Gradient Boosting Classification
**Prerequisites**: Iris data uploaded
**Command**: `train gradient boosting model to predict species`

**Expected Response**:
```
Model Type: gradient_boosting
Training Metrics:
• Accuracy: 0.93-1.00
• Learning Rate: 0.1 (default)
• Number of Estimators: 100
```

**Success Criteria**: Same as Test 3.6

---

#### Test 3.10: SVM Classification
**Prerequisites**: Iris data uploaded
**Command**: `use svm to classify species`

**Expected Response**:
```
Model Type: svm
Training Metrics:
• Accuracy: 0.90-1.00
• Kernel: rbf (default)
• Support Vectors identified
```

**Success Criteria**: Same as Test 3.6
**Note**: May take slightly longer to train

---

#### Test 3.11: Naive Bayes Classification
**Prerequisites**: Iris data uploaded
**Command**: `train naive bayes for species prediction`

**Expected Response**:
```
Model Type: naive_bayes
Training Metrics:
• Accuracy: 0.90-0.98
• Probabilistic classification
• Fast training time
```

**Success Criteria**: Same as Test 3.6

---

#### Test 3.12: MLP Regression (Neural Network)
**Prerequisites**: Housing data uploaded
**Command**: `train neural network for price regression`

**Expected Response**:
```
✅ Model Training Complete

Model ID: model_12345_mlp_regression
Task Type: neural_network
Model Type: mlp_regression

Training Metrics:
• R² Score: 0.75-0.90
• RMSE: 2,000-3,500
• MAE: 1,800-2,800

Neural Network Architecture:
• Hidden Layers: [100] (default)
• Activation: relu
• Max Iterations: 200
• May show convergence warnings (normal)

Training Time: 2-5s (longer than traditional models)
```

**Success Criteria**:
- ✅ Model trains (may show convergence warning - OK)
- ✅ R² > 0.70
- ✅ Architecture information provided
- ✅ Training time < 10 seconds

**Note**: May see convergence warnings - this is expected with small datasets and default iterations

---

#### Test 3.13: MLP Classification (Neural Network)
**Prerequisites**: Iris data uploaded
**Command**: `train neural network classifier for species`

**Expected Response**:
```
Model Type: mlp_classification
Training Metrics:
• Accuracy: 0.85-1.00
• Hidden Layers: [100]
• Activation: relu
• May show convergence warnings

Training Time: 2-5s
```

**Success Criteria**:
- ✅ Accuracy > 0.80
- ✅ Multi-class classification working
- ✅ Convergence warnings OK

---

### Phase 4: ML Predictions and Model Management

#### Test 4.1: Making Predictions
**Prerequisites**:
1. Housing data uploaded
2. Model trained (e.g., random_forest from Test 3.4)
3. New data file created: `new_houses.csv`

**New Data File** (`new_houses.csv`):
```csv
sqft,bedrooms,bathrooms,age
1350,2,1,8
2100,4,3,4
1650,3,2,6
```

**Steps**:
1. Upload `new_houses.csv`
2. Send command: `predict prices using model model_12345_random_forest`

**Expected Response**:
```
✅ Predictions Complete

Model ID: model_12345_random_forest
Model Type: random_forest

Predictions:
1. Row 1: $265,000
2. Row 2: $405,000
3. Row 3: $310,000

Prediction Statistics:
• Count: 3
• Mean: $326,667
• Min: $265,000
• Max: $405,000

Model Info:
• Task Type: regression
• Features Used: sqft, bedrooms, bathrooms, age
• Training R²: 0.87

Execution Time: 0.3s
```

**Success Criteria**:
- ✅ Predictions returned for all rows
- ✅ Values are reasonable (within training data range)
- ✅ Statistics calculated correctly
- ✅ Model info displayed
- ✅ Fast execution (< 1 second)

**Alternative Commands**:
- `score new data with model_12345_random_forest`
- `use model model_12345_random_forest to predict`

---

#### Test 4.2: Classification Predictions
**Prerequisites**:
1. Iris model trained (e.g., from Test 3.8)
2. New data file: `new_iris.csv`

**New Data File** (`new_iris.csv`):
```csv
sepal_length,sepal_width,petal_length,petal_width
5.0,3.5,1.3,0.3
6.5,3.0,5.2,2.0
5.9,3.0,4.2,1.5
```

**Command**: `predict species using model model_12345_random_forest`

**Expected Response**:
```
✅ Predictions Complete

Predictions:
1. Row 1: setosa (confidence: 100%)
2. Row 2: virginica (confidence: 95%)
3. Row 3: versicolor (confidence: 92%)

Class Probabilities:
Row 1: setosa=1.00, versicolor=0.00, virginica=0.00
Row 2: setosa=0.00, versicolor=0.05, virginica=0.95
Row 3: setosa=0.02, versicolor=0.92, virginica=0.06

Prediction Statistics:
• Total Predictions: 3
• Classes: setosa, versicolor, virginica

Execution Time: 0.2s
```

**Success Criteria**:
- ✅ Class labels predicted
- ✅ Probabilities sum to 1.0 per row
- ✅ Confidence scores provided
- ✅ All target classes represented

---

#### Test 4.3: List User Models
**Prerequisites**: At least 2 models trained
**Command**: `list my models` or `show all my trained models`

**Expected Response**:
```
📋 Your Trained Models

1. model_12345_random_forest
   • Type: regression / random_forest
   • Trained: 2025-01-28 14:35:22
   • Features: 4
   • R² Score: 0.87
   • Size: 2.3 MB

2. model_12346_logistic
   • Type: classification / logistic
   • Trained: 2025-01-28 14:42:15
   • Features: 4
   • Accuracy: 0.95
   • Size: 0.8 MB

Total Models: 2
Storage Used: 3.1 MB / 5000 MB
```

**Success Criteria**:
- ✅ All trained models listed
- ✅ Metadata displayed for each
- ✅ Timestamps correct
- ✅ Storage usage shown

---

#### Test 4.4: Get Model Info
**Prerequisites**: At least one model trained
**Command**: `show info for model model_12345_random_forest`

**Expected Response**:
```
📊 Model Information

Model ID: model_12345_random_forest
Status: Active

Configuration:
• Task Type: regression
• Model Type: random_forest
• Target Column: price
• Feature Columns: sqft, bedrooms, bathrooms, age

Performance Metrics:
• R² Score: 0.87
• RMSE: 2,156
• MAE: 1,842

Training Details:
• Samples: 20 (16 train, 4 test)
• Training Time: 1.2s
• Trained: 2025-01-28 14:35:22

Hyperparameters:
• n_estimators: 100
• max_depth: None
• min_samples_split: 2

Preprocessing:
• Missing Strategy: mean
• Scaling: standard

Storage:
• Model Size: 2.3 MB
• Location: models/user_12345/model_12345_random_forest/
```

**Success Criteria**:
- ✅ Complete model details displayed
- ✅ All metrics present
- ✅ Configuration accurate
- ✅ Preprocessing info shown

---

### Phase 5: Error Handling Tests

#### Test 5.1: No Data Uploaded
**Command**: `train a model` (without uploading data first)

**Expected Response**:
```
⚠️ No Data Available

Please upload a CSV file first before requesting analysis.

To upload data:
1. Click the paperclip/attachment icon
2. Select your CSV file
3. Send it to me

Once I receive your data, I'll confirm the upload and you can
request any analysis you need!
```

**Success Criteria**:
- ✅ Clear error message
- ✅ Instructions provided
- ✅ No crash or unexpected behavior

---

#### Test 5.2: Invalid Column Reference
**Prerequisites**: Housing data uploaded
**Command**: `train a model to predict sale_price` (column doesn't exist)

**Expected Response**:
```
❌ Validation Error

Column 'sale_price' not found in dataset.

Available columns:
• sqft
• bedrooms
• bathrooms
• age
• price

Did you mean 'price'?
```

**Success Criteria**:
- ✅ Error caught gracefully
- ✅ Available columns listed
- ✅ Helpful suggestion if applicable

---

#### Test 5.3: Insufficient Data
**Prerequisites**: Upload CSV with only 3 rows
**Command**: `train a model to predict price`

**Expected Response**:
```
❌ Data Validation Error

Insufficient training data: 3 samples found.
Minimum required: 10 samples for reliable model training.

Please upload a larger dataset with at least 10 rows.
```

**Success Criteria**:
- ✅ Minimum sample check enforced
- ✅ Clear error message
- ✅ Requirement stated

---

#### Test 5.4: Invalid Model Type
**Prerequisites**: Housing data uploaded
**Command**: `train a super_model to predict price` (invalid model name)

**Expected Response**:
```
❌ Validation Error

Model type 'super_model' not supported for task 'regression'.

Supported regression models:
• linear - Linear Regression
• ridge - Ridge Regression
• lasso - Lasso Regression
• elasticnet - ElasticNet
• polynomial - Polynomial Regression
• random_forest - Random Forest
• gradient_boosting - Gradient Boosting
• mlp_regression - Neural Network

Example: "train a random_forest model to predict price"
```

**Success Criteria**:
- ✅ Invalid model detected
- ✅ All valid options listed
- ✅ Example provided

---

#### Test 5.5: Prediction with Non-existent Model
**Command**: `predict using model model_99999_fake`

**Expected Response**:
```
❌ Model Not Found

Model 'model_99999_fake' does not exist or was deleted.

Your available models:
• model_12345_random_forest (regression)
• model_12346_logistic (classification)

Use 'list my models' to see all your trained models.
```

**Success Criteria**:
- ✅ Clear error message
- ✅ Available models suggested
- ✅ Helpful command provided

---

## 🎯 Success Metrics

### Overall Testing Success Criteria

**Basic Functionality**: All 4 tests pass
- ✅ Bot responds to commands
- ✅ File upload works
- ✅ Data inspection works
- ✅ Help/start commands functional

**Statistics**: Both tests pass
- ✅ Descriptive statistics calculated correctly
- ✅ Correlation analysis works

**ML Training**: At least 10/13 model tests pass
- ✅ Regression models train successfully (5 models)
- ✅ Classification models train successfully (6 models)
- ✅ Neural networks train (may have convergence warnings - OK)
- ✅ All models save and return model IDs

**ML Operations**: All 4 tests pass
- ✅ Predictions work for regression and classification
- ✅ Model listing functional
- ✅ Model info retrieval works
- ✅ Model management operational

**Error Handling**: At least 4/5 tests pass
- ✅ Graceful error messages
- ✅ Validation catches issues
- ✅ Helpful error information provided

---

## 🚨 Troubleshooting

### Bot Doesn't Respond
**Check**:
- Bot is running (`python src/bot/telegram_bot.py`)
- .env file has valid TELEGRAM_BOT_TOKEN
- Check console for error messages
- Try /start command to verify bot is online

### File Upload Fails
**Check**:
- File is actually CSV format (not Excel, not text)
- File size < 100MB
- File has headers in first row
- File is properly formatted (no extra blank lines at top)

### Model Training Fails
**Check**:
- Data has been uploaded first
- Column names are correct (check with "what columns")
- At least 10 rows of data
- Numeric columns for regression targets
- Categorical/binary columns for classification targets

### Predictions Fail
**Check**:
- Model ID is correct (use "list my models")
- New data has same column names as training data
- New data is uploaded before prediction command
- Feature columns match training data

### Low Model Accuracy
**Expected**:
- Small datasets (< 50 rows) may have lower accuracy
- Neural networks may show convergence warnings - this is normal
- Some models work better for certain data types

**Tips**:
- Ensure data quality (no missing values ideally)
- Use more data for better results
- Try different model types
- Check that target variable is appropriate

---

## 📊 Performance Benchmarks

### Expected Response Times
- Bot commands (/start, /help): < 0.5s
- File upload processing: 1-3s for < 100KB
- Descriptive statistics: < 1s
- Correlation analysis: < 2s
- Linear model training: 1-3s
- Tree-based model training: 2-5s
- Neural network training: 3-10s
- Predictions: < 1s
- Model listing: < 0.5s

### Expected Accuracy Ranges
**Regression Models**:
- R² > 0.7 = Good
- R² > 0.8 = Very Good
- R² > 0.9 = Excellent (may indicate overfitting on small data)

**Classification Models**:
- Accuracy > 0.8 = Good
- Accuracy > 0.9 = Very Good
- Accuracy > 0.95 = Excellent

**Note**: Small datasets (< 100 samples) may show inflated or variable metrics

---

## 📝 Testing Log Template

Use this template to track your testing:

```
Date: _____________
Tester: _____________
Bot Version: _____________

PHASE 1: BASIC FUNCTIONALITY
[ ] Test 1.1 - Bot Startup: _____ (Pass/Fail)
[ ] Test 1.2 - Help Command: _____ (Pass/Fail)
[ ] Test 1.3 - File Upload: _____ (Pass/Fail)
[ ] Test 1.4 - Data Inspection: _____ (Pass/Fail)

PHASE 2: STATISTICS
[ ] Test 2.1 - Descriptive Stats: _____ (Pass/Fail)
[ ] Test 2.2 - Correlation: _____ (Pass/Fail)

PHASE 3: ML TRAINING
[ ] Test 3.1 - Linear Regression: _____ (Pass/Fail)
[ ] Test 3.2 - Ridge Regression: _____ (Pass/Fail)
[ ] Test 3.3 - Lasso Regression: _____ (Pass/Fail)
[ ] Test 3.4 - Random Forest Reg: _____ (Pass/Fail)
[ ] Test 3.5 - Polynomial Reg: _____ (Pass/Fail)
[ ] Test 3.6 - Logistic Regression: _____ (Pass/Fail)
[ ] Test 3.7 - Decision Tree: _____ (Pass/Fail)
[ ] Test 3.8 - Random Forest Class: _____ (Pass/Fail)
[ ] Test 3.9 - Gradient Boosting: _____ (Pass/Fail)
[ ] Test 3.10 - SVM: _____ (Pass/Fail)
[ ] Test 3.11 - Naive Bayes: _____ (Pass/Fail)
[ ] Test 3.12 - MLP Regression: _____ (Pass/Fail)
[ ] Test 3.13 - MLP Classification: _____ (Pass/Fail)

PHASE 4: ML OPERATIONS
[ ] Test 4.1 - Predictions (Regression): _____ (Pass/Fail)
[ ] Test 4.2 - Predictions (Classification): _____ (Pass/Fail)
[ ] Test 4.3 - List Models: _____ (Pass/Fail)
[ ] Test 4.4 - Model Info: _____ (Pass/Fail)

PHASE 5: ERROR HANDLING
[ ] Test 5.1 - No Data Error: _____ (Pass/Fail)
[ ] Test 5.2 - Invalid Column: _____ (Pass/Fail)
[ ] Test 5.3 - Insufficient Data: _____ (Pass/Fail)
[ ] Test 5.4 - Invalid Model: _____ (Pass/Fail)
[ ] Test 5.5 - Model Not Found: _____ (Pass/Fail)

OVERALL RESULTS:
Tests Passed: ____ / 28
Tests Failed: ____
Success Rate: ____%

NOTES:
_________________________________
_________________________________
_________________________________
```

---

## 🎓 Tips for Effective Testing

### 1. Test Incrementally
- Start with Phase 1 (basic functionality)
- Only proceed to next phase if previous passes
- Don't skip error handling tests

### 2. Document Everything
- Screenshot unexpected behaviors
- Note exact commands that fail
- Record error messages verbatim
- Track response times

### 3. Test Variations
- Try different phrasing for same commands
- Test with different data sizes
- Try edge cases (1 feature, many features)
- Test with real-world messy data

### 4. Verify Results
- Check if predictions make sense
- Verify statistics against Excel/calculator
- Ensure model IDs are consistent
- Confirm timestamps are accurate

### 5. Clean State Testing
- Start fresh session for critical tests
- Clear bot data between test phases if needed
- Use different datasets for different model types
- Verify models persist across bot restarts

---

## ✅ Final Checklist

Before reporting "All Tests Passed":

- [ ] All 5 phases completed
- [ ] At least 24/28 tests passed (85%+)
- [ ] All critical paths work (upload → train → predict)
- [ ] Error handling graceful (no crashes)
- [ ] Performance within acceptable ranges
- [ ] Documentation complete
- [ ] Issues logged with details
- [ ] Screenshots captured for key successes

**Congratulations! Your ML Engine is fully operational! 🎉**
