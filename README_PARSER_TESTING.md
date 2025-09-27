# Parser Testing with Real Telegram Messages

## Quick Start

1. **Run the Parser Test Bot:**
```bash
./start_parser_test_bot.sh
```

2. **Go to your Telegram bot and try these commands:**

### Basic Commands
- `/start` - Welcome and instructions
- `/help` - Detailed help with examples
- `/debug` - Show parser technical details
- `/data` - Show uploaded file information

### Test Messages

**📊 Statistical Requests:**
- `calculate mean`
- `show correlation between age and income`
- `descriptive statistics for all columns`
- `standard deviation of salary`

**🧠 Machine Learning Requests:**
- `train a model to predict income`
- `build random forest classifier`
- `predict house prices using regression`
- `neural network for customer satisfaction`

**📋 Data Information:**
- `show me the data`
- `what columns are available`
- `data shape and size`

**⚠️ Edge Cases:**
- `CALCULATE MEAN` (test case sensitivity)
- `calc mean 4 age` (test typos)
- `do something with data` (test ambiguous requests)

## What You'll See

The test bot shows detailed parsing results for every message:

```
✅ Parsing Successful! 🟢

Original: "calculate mean for age"

📊 Task Type: stats
🎯 Operation: Mean Analysis
📈 Confidence: 80%

📋 Parameters:
• Statistics: mean
• Columns: age

🚀 Next Steps:
• Route to Statistics Engine
• Generate analysis script
• Calculate: mean
```

## Features

### 🎯 **Real-time Parser Testing**
- See exactly how your natural language is parsed
- View confidence scores and extracted parameters
- Test different phrasings and variations

### 📁 **File Upload Testing**
- Upload CSV files to test data context
- Parser uses file metadata for better accuracy
- Mock columns: age, income, education, satisfaction, region

### 🔧 **Debug Information**
- `/debug` shows loaded patterns and thresholds
- Detailed error messages for failed parsing
- Logs all parsing attempts for analysis

### 📊 **Comprehensive Feedback**
- Shows what engine would handle the request
- Displays extracted columns, targets, and features
- Explains next steps in the processing pipeline

## Testing Strategy

1. **Upload a CSV file first** - This gives the parser data context
2. **Try basic requests** - Start with simple commands
3. **Test variations** - Try different ways to say the same thing
4. **Test edge cases** - Try ambiguous or unusual requests
5. **Check confidence scores** - Lower scores indicate parsing uncertainty

## Interpreting Results

### Confidence Levels
- 🟢 **70%+**: High confidence, clear understanding
- 🟡 **40-69%**: Medium confidence, likely correct
- 🔴 **<40%**: Low confidence, may need clarification

### Task Types
- 📊 **stats**: Statistical analysis requests
- 🧠 **ml_train**: Model training requests
- 🔮 **ml_score**: Prediction requests
- 📋 **data_info**: Data exploration requests

## Common Test Scenarios

### ✅ Should Work Well
- "calculate mean for age column"
- "show correlation matrix"
- "train model to predict income"
- "what columns are in the data"

### ⚠️ May Need Improvement
- Very long, complex requests
- Requests mixing multiple operations
- Non-English words or heavy slang
- Very ambiguous requests

### ❌ Expected to Fail
- Empty messages
- Random text/gibberish
- Requests for unsupported operations

## Next Steps

After testing with this bot:

1. **Identify patterns that need improvement**
2. **Note common user phrasings not handled well**
3. **Test the improved patterns**
4. **Integrate parser into main bot**

The parser test bot gives you real-world validation before deploying to production!