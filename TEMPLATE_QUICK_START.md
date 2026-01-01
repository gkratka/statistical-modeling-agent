# Template System Quick Start Guide

## What are Templates?

Templates let you save ML workflow configurations and reuse them with a single command. Perfect for:
- Repeating experiments with same setup
- Standardizing workflows across datasets
- Quick access to common model configurations

## Template Naming

**Format**: `TRAIN_<MODEL>_<TARGET>` or `PREDICT_<MODEL>_<COLUMN>`
- Model/Target/Column: 1-8 uppercase alphanumeric characters
- Examples:
  - `TRAIN_CATBST_CLASS2` - Train CatBoost for class2 target
  - `PREDICT_XGBST_PRICE` - Predict with XGBoost for price column
  - `TRAIN_KERAS_CHURN` - Train Keras model for churn prediction

## Quick Commands

```bash
# Show help and your templates
/template

# Run a template (shows configuration)
/template TRAIN_CATBST_CLASS2

# List all templates with details
/template list

# Delete a template
/template delete TRAIN_CATBST_CLASS2
```

## Creating Templates

### For Training Workflows
1. Complete `/train` workflow
2. After training completes, click "💾 Save as Template"
3. Enter template name (e.g., `TRAIN_CATBST_CLASS2`)
4. Done! Use `/template TRAIN_CATBST_CLASS2` to view config

### For Prediction Workflows
1. Complete `/predict` workflow
2. After predictions complete, click "💾 Save as Template"
3. Enter template name (e.g., `PREDICT_CATBST_CLASS2`)
4. Done! Use `/template PREDICT_CATBST_CLASS2` to view config

## What Gets Saved?

**Training Templates**:
- Data file path
- Target column
- Feature columns
- Model type
- Hyperparameters
- Model name (if set)

**Prediction Templates**:
- Data file path
- Model ID
- Feature columns
- Output column name
- Output file path (if set)

## Examples

### Save Training Template
```
You: /train
Bot: [Complete training workflow...]
Bot: Training complete! Model: catboost_binary_classification
     Buttons: [Name Model] [Skip] [💾 Save as Template]
You: [Click "Save as Template"]
Bot: Enter template name (format: TRAIN_XXXXXXXX_XXXXXXXX)
You: TRAIN_CATBST_CLASS2
Bot: ✅ Template 'TRAIN_CATBST_CLASS2' saved!
     Use /template TRAIN_CATBST_CLASS2 to run.
```

### Use Template
```
You: /template TRAIN_CATBST_CLASS2
Bot: ✅ Template Configuration Loaded
     📁 Data: /path/to/data.csv
     🎯 Target: class2
     📊 Features: col1, col2, col3
     🤖 Model: catboost_binary_classification

     Next Steps:
     1. Use /train to start training workflow
     2. Select 'Use Template' → TRAIN_CATBST_CLASS2
```

### List Templates
```
You: /template list
Bot: 📋 Your Templates (5 total)
     • TRAIN: 3
     • PREDICT: 2

     TRAIN Templates:
     • TRAIN_CATBST_CLASS2 - Target: class2, Model: catboost (Created: 2024-12-29)
     • TRAIN_XGBST_PRICE - Target: price, Model: xgboost (Created: 2024-12-28)

     PREDICT Templates:
     • PREDICT_CATBST_CLASS2 - Model: model_123... (Created: 2024-12-29)
```

## Tips & Best Practices

1. **Use Descriptive Names**: Include model type and target in name
   - Good: `TRAIN_CATBST_CLASS2`
   - Bad: `TRAIN_MODEL1_COL1`

2. **Organize by Project**: Use consistent naming for related templates
   - `TRAIN_HOUSING_PRICE`, `PREDICT_HOUSING_PRICE`

3. **Keep Templates Updated**: Delete old templates when file paths change
   - `/template delete OLD_TEMPLATE`

4. **Check Template Before Use**: Verify file paths still exist
   - Template execution will warn if file path is invalid

5. **Template Limit**: Max 50 templates per user
   - Delete unused templates to stay under limit

## Troubleshooting

### "Invalid template name" Error
**Problem**: Name doesn't match required format
**Solution**: Use format `TRAIN_XXXXXXXX_XXXXXXXX` or `PREDICT_XXXXXXXX_XXXXXXXX`
- Segments must be 1-8 uppercase alphanumeric
- Example: `TRAIN_CATBST_CLASS2` ✅
- Wrong: `train_catbst_class2` ❌ (lowercase)
- Wrong: `TRAIN_CATBOOST_CLASS2` ❌ (CATBOOST > 8 chars)

### "Template already exists" Error
**Problem**: Template name is already taken
**Solution**: Choose different name or delete existing template
```bash
# Delete old template
/template delete TRAIN_CATBST_CLASS2

# Or use different name
TRAIN_CATBST_CLASS3
```

### "Data file not found" Error
**Problem**: File path in template no longer exists
**Solution**:
- Verify file location
- Re-save template with updated path
- Or use different template

### Template Not Appearing in List
**Problem**: Template saved but not showing
**Solution**:
- Send `/template` to refresh list
- Check if saved in correct format
- Verify you're using correct user account

## Storage Location

Templates are stored in: `./templates/<user_id>.json`
- One JSON file per user
- All templates in single file
- Auto-created on first save

## Next Steps

After reviewing template configuration with `/template <name>`:
1. Use `/train` to start training with these settings
2. Or use `/predict` to run predictions
3. Select "Use Template" option in workflow (if available)
4. Or manually enter the same configuration

## Feedback & Support

Having issues with templates?
- Use `/help` for general bot commands
- Check template format with `/template`
- Verify file paths are accessible
- Contact support if problems persist
