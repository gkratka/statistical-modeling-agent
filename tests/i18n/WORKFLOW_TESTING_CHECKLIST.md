# i18n Workflow Testing Checklist

## Overview
This checklist ensures comprehensive i18n coverage across all user-facing workflows in both English (`/en`) and Portuguese (`/pt`) modes.

## Testing Instructions
1. Start fresh conversation with bot
2. Set language mode with `/pt` or `/en`
3. Complete entire workflow verifying NO hardcoded English strings appear
4. Check all: messages, buttons, confirmations, errors, success messages
5. Mark checkbox with `[x]` when workflow fully passes in both languages

---

## ML Training Workflows

### Regression Models

#### [ ] Linear Regression (/pt mode)
- [ ] `/train` command response
- [ ] Data source selection (Telegram Upload vs Local Path)
- [ ] Schema detection display
- [ ] Column list with stats
- [ ] Suggested target/features display
- [ ] Schema confirmation prompt
- [ ] Target column selection
- [ ] Feature column selection
- [ ] Training start message ("Starting training...")
- [ ] Training progress indicator ("This may take a few moments.")
- [ ] Model completion message ("Model Ready!")
- [ ] Default model name display ("Default Name: ...")
- [ ] Model ID display ("Model ID: ...")
- [ ] Model type display ("Type: ...")
- [ ] Ready for predictions message
- [ ] Model naming prompt
- [ ] Template save option
- [ ] Final confirmation

#### [ ] Linear Regression (/en mode)
- [ ] Same checklist as above

#### [ ] Ridge Regression (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (alpha)

#### [ ] Ridge Regression (/en mode)
- [ ] Complete workflow as above

#### [ ] Lasso Regression (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (alpha)

#### [ ] Lasso Regression (/en mode)
- [ ] Complete workflow as above

#### [ ] ElasticNet Regression (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (alpha, l1_ratio)

#### [ ] ElasticNet Regression (/en mode)
- [ ] Complete workflow as above

#### [ ] Polynomial Regression (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (degree)

#### [ ] Polynomial Regression (/en mode)
- [ ] Complete workflow as above

### Classification Models

#### [ ] Logistic Regression (/pt mode)
- [ ] Complete workflow as regression
- [ ] Binary vs multi-class detection
- [ ] Class imbalance warnings (if applicable)

#### [ ] Logistic Regression (/en mode)
- [ ] Complete workflow as above

#### [ ] Decision Tree Classifier (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (max_depth, min_samples_split)

#### [ ] Decision Tree Classifier (/en mode)
- [ ] Complete workflow as above

#### [ ] Random Forest Classifier (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (n_estimators, max_depth)

#### [ ] Random Forest Classifier (/en mode)
- [ ] Complete workflow as above

#### [ ] Gradient Boosting Classifier (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (n_estimators, learning_rate)

#### [ ] Gradient Boosting Classifier (/en mode)
- [ ] Complete workflow as above

#### [ ] SVM Classifier (/pt mode)
- [ ] Complete workflow as above
- [ ] Hyperparameter configuration (C, kernel)

#### [ ] SVM Classifier (/en mode)
- [ ] Complete workflow as above

#### [ ] Naive Bayes Classifier (/pt mode)
- [ ] Complete workflow as above

#### [ ] Naive Bayes Classifier (/en mode)
- [ ] Complete workflow as above

### Neural Network Models

#### [ ] Keras Binary Classification (/pt mode)
- [ ] Complete workflow as classification
- [ ] Architecture configuration (layers, neurons)
- [ ] Training parameters (epochs, batch_size)
- [ ] Validation split display
- [ ] Training progress updates
- [ ] **CRITICAL**: "Starting training..." message
- [ ] **CRITICAL**: "This may take a few moments." message
- [ ] **CRITICAL**: "Model Ready!" message
- [ ] **CRITICAL**: "Default Name: Binary Classification - Nov 20 2025" format
- [ ] **CRITICAL**: "Model ID: ..." display
- [ ] **CRITICAL**: "Type: kerasbinaryclassification" display
- [ ] **CRITICAL**: "ready for predictions!" message

#### [ ] Keras Binary Classification (/en mode)
- [ ] Complete workflow as above

#### [ ] Keras Multi-Class Classification (/pt mode)
- [ ] Complete workflow as above
- [ ] Number of classes detection
- [ ] Softmax activation display

#### [ ] Keras Multi-Class Classification (/en mode)
- [ ] Complete workflow as above

#### [ ] Keras Regression (/pt mode)
- [ ] Complete workflow as above
- [ ] Linear activation display

#### [ ] Keras Regression (/en mode)
- [ ] Complete workflow as above

---

## Data Source Workflows

### Telegram Upload

#### [ ] CSV Upload (/pt mode)
- [ ] File upload prompt
- [ ] Processing message
- [ ] Success confirmation
- [ ] Preview display
- [ ] Column type detection
- [ ] Row count display

#### [ ] CSV Upload (/en mode)
- [ ] Complete workflow as above

#### [ ] Excel Upload (/pt mode)
- [ ] Same as CSV workflow
- [ ] Sheet selection (if multiple)

#### [ ] Excel Upload (/en mode)
- [ ] Complete workflow as above

#### [ ] Parquet Upload (/pt mode)
- [ ] Same as CSV workflow

#### [ ] Parquet Upload (/en mode)
- [ ] Complete workflow as above

### Local Path Upload

#### [ ] Valid Path (/pt mode)
- [ ] Path input prompt
- [ ] Validation messages
- [ ] Security check confirmation
- [ ] File loading message
- [ ] Success confirmation

#### [ ] Valid Path (/en mode)
- [ ] Complete workflow as above

#### [ ] Invalid Path - Security (/pt mode)
- [ ] Path traversal detection error
- [ ] Whitelist violation error
- [ ] Symlink resolution error

#### [ ] Invalid Path - Security (/en mode)
- [ ] Complete workflow as above

#### [ ] Invalid Path - Validation (/pt mode)
- [ ] File not found error
- [ ] Invalid extension error
- [ ] File too large error
- [ ] Empty file error

#### [ ] Invalid Path - Validation (/en mode)
- [ ] Complete workflow as above

---

## Schema Detection

#### [ ] Auto-Detection Success (/pt mode)
- [ ] Dataset statistics display
- [ ] Target column suggestion
- [ ] Feature columns suggestion
- [ ] Task type suggestion (regression/classification)
- [ ] Confidence indicators
- [ ] Accept/Reject buttons

#### [ ] Auto-Detection Success (/en mode)
- [ ] Complete workflow as above

#### [ ] Auto-Detection Partial (/pt mode)
- [ ] Uncertain suggestions display
- [ ] Manual selection prompt
- [ ] Recommendation explanations

#### [ ] Auto-Detection Partial (/en mode)
- [ ] Complete workflow as above

#### [ ] Auto-Detection Failure (/pt mode)
- [ ] Error message
- [ ] Manual selection fallback
- [ ] Column list display

#### [ ] Auto-Detection Failure (/en mode)
- [ ] Complete workflow as above

---

## Parameter Configuration

#### [ ] Default Parameters (/pt mode)
- [ ] "Use defaults" button text
- [ ] Default values display
- [ ] Confirmation message

#### [ ] Default Parameters (/en mode)
- [ ] Complete workflow as above

#### [ ] Custom Parameters (/pt mode)
- [ ] "Configure" button text
- [ ] Parameter input prompts
- [ ] Validation messages
- [ ] Range limit errors
- [ ] Type conversion errors

#### [ ] Custom Parameters (/en mode)
- [ ] Complete workflow as above

---

## Training Execution

#### [ ] Training Start (/pt mode)
- [ ] **CRITICAL**: "Starting training..." message
- [ ] **CRITICAL**: "This may take a few moments." message
- [ ] Progress indicator (if any)

#### [ ] Training Start (/en mode)
- [ ] Complete workflow as above

#### [ ] Training Success (/pt mode)
- [ ] **CRITICAL**: "Model Ready!" header
- [ ] Metrics display (R², MSE, Accuracy, etc.)
- [ ] Training time display
- [ ] Model info section
- [ ] Next steps guidance

#### [ ] Training Success (/en mode)
- [ ] Complete workflow as above

#### [ ] Training Failure (/pt mode)
- [ ] Error message
- [ ] Reason explanation
- [ ] Retry guidance
- [ ] Data quality suggestions

#### [ ] Training Failure (/en mode)
- [ ] Complete workflow as above

---

## Model Completion & Naming

#### [ ] Default Name Display (/pt mode)
- [ ] **CRITICAL**: "Default Name: [Type] - [Date]" format
- [ ] **CRITICAL**: "Model ID: ..." display
- [ ] **CRITICAL**: "Type: ..." display
- [ ] **CRITICAL**: "ready for predictions!" message
- [ ] Custom name prompt
- [ ] "Keep default" button text

#### [ ] Default Name Display (/en mode)
- [ ] Complete workflow as above

#### [ ] Custom Name Input (/pt mode)
- [ ] Name input prompt
- [ ] Validation messages
- [ ] Length limit errors
- [ ] Special character warnings
- [ ] Confirmation message

#### [ ] Custom Name Input (/en mode)
- [ ] Complete workflow as above

---

## Template Saving

#### [ ] Save Template Offer (/pt mode)
- [ ] "Save as template?" prompt
- [ ] Template name input
- [ ] Description input
- [ ] Reuse explanation
- [ ] Success confirmation

#### [ ] Save Template Offer (/en mode)
- [ ] Complete workflow as above

#### [ ] Template Reuse (/pt mode)
- [ ] Template list display
- [ ] Selection buttons
- [ ] Applied parameters display
- [ ] Confirmation message

#### [ ] Template Reuse (/en mode)
- [ ] Complete workflow as above

---

## Prediction Workflow

#### [ ] Model Selection (/pt mode)
- [ ] `/predict` command response
- [ ] User model list display
- [ ] Model type indicators
- [ ] Selection buttons

#### [ ] Model Selection (/en mode)
- [ ] Complete workflow as above

#### [ ] Data Upload for Prediction (/pt mode)
- [ ] Upload prompt
- [ ] Schema validation
- [ ] Feature mismatch errors
- [ ] Processing message

#### [ ] Data Upload for Prediction (/en mode)
- [ ] Complete workflow as above

#### [ ] Prediction Success (/pt mode)
- [ ] Results display
- [ ] Confidence scores (if classification)
- [ ] Download option
- [ ] Visualization option

#### [ ] Prediction Success (/en mode)
- [ ] Complete workflow as above

#### [ ] Prediction Failure (/pt mode)
- [ ] Error message
- [ ] Feature mismatch details
- [ ] Data format guidance

#### [ ] Prediction Failure (/en mode)
- [ ] Complete workflow as above

---

## Error Handling

#### [ ] Validation Errors (/pt mode)
- [ ] Invalid column selection
- [ ] Invalid parameter values
- [ ] File format errors
- [ ] Data quality errors

#### [ ] Validation Errors (/en mode)
- [ ] Complete workflow as above

#### [ ] System Errors (/pt mode)
- [ ] Training timeout
- [ ] Memory limit exceeded
- [ ] Model save failure
- [ ] Generic error fallback

#### [ ] System Errors (/en mode)
- [ ] Complete workflow as above

#### [ ] User Errors (/pt mode)
- [ ] Invalid command in workflow
- [ ] Missing required data
- [ ] Cancelled operation

#### [ ] User Errors (/en mode)
- [ ] Complete workflow as above

---

## Language Switching

#### [ ] Mid-Workflow Switch /pt → /en
- [ ] Language change confirmation
- [ ] State preservation
- [ ] Next message in new language

#### [ ] Mid-Workflow Switch /en → /pt
- [ ] Complete workflow as above

#### [ ] Persistence Across Sessions
- [ ] User preference storage
- [ ] New session respects preference
- [ ] Override with explicit command

---

## Testing Statistics

- **Total Test Cases**: 200+
- **Critical String Issues**: 7 (from screenshot)
- **Languages**: 2 (English, Portuguese)
- **Model Types**: 13
- **Workflows**: 8 major workflows

---

## Known Issues (To Fix)

### From Screenshot Analysis:
1. ❌ "🚀 Starting training..." - needs i18n
2. ❌ "This may take a few moments." - needs i18n
3. ❌ "✅ Model Ready!" - needs i18n
4. ❌ "📋 Default Name: Binary Classification - Nov 20 2025" - needs i18n
5. ❌ "🆔 Model ID:" - needs i18n
6. ❌ "🎯 Type: kerasbinaryclassification" - needs i18n
7. ❌ "📋 Your model is ready for predictions!" - needs i18n

---

## Testing Best Practices

1. **Fresh Start**: Always start with `/start` to reset state
2. **Clear Language**: Explicitly set `/pt` or `/en` at start
3. **Complete Flows**: Test entire workflow, not just parts
4. **Edge Cases**: Test invalid inputs, cancellations, errors
5. **Screenshots**: Capture any remaining English strings in /pt mode
6. **Documentation**: Record any issues in GitHub issues

---

## Completion Criteria

- [ ] All regression models tested in both languages
- [ ] All classification models tested in both languages
- [ ] All neural network models tested in both languages
- [ ] All data source workflows tested
- [ ] All error cases tested
- [ ] Zero hardcoded English strings in /pt mode
- [ ] Language switching works mid-workflow
- [ ] User preferences persist across sessions

---

**Last Updated**: 2025-11-19
**Next Review**: After fixing screenshot issues
