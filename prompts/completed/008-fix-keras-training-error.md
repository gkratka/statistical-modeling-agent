<objective>
Fix the Keras binary classification training error: `ValueError: Cannot convert '(20, 'layers')' to a shape. Found invalid entry 'layers' of type '<class 'str'>'`
</objective>

<context>
Critical bug: Keras model training fails with input_shape construction error.

Error message:
```
ValueError: Cannot convert '(20, 'layers')' to a shape.
Found invalid entry 'layers' of type '<class 'str'>'.
```

Root cause analysis:
- The `input_shape` tuple is incorrectly constructed as `(20, 'layers')`
- Should be `(20,)` where 20 is the number of input features
- The string 'layers' is being incorrectly included in the shape tuple
- This likely happens when parsing hyperparameters or building the model

@src/engines/ml_engine.py - ML engine
@src/bot/ml_handlers/ - Keras training handlers
</context>

<research>
1. Search for "input_shape" in the codebase
2. Find where Keras models are built
3. Look for where hyperparameters are parsed and passed to model builder
4. Find where 'layers' might be getting mixed into the shape
</research>

<requirements>
1. Fix input_shape to be proper tuple like `(n_features,)`
2. Ensure 'layers' configuration is separate from shape
3. Keras binary classification must train successfully
</requirements>

<implementation>
1. Find the Keras model building code
2. Locate where input_shape is constructed
3. Fix the tuple construction - likely needs to be:
   ```python
   # Wrong: (n_features, 'layers') or (n_features, something_else)
   # Right: (n_features,)
   input_shape = (X_train.shape[1],)  # Just the number of features
   ```
4. Ensure hyperparameters dict keys like 'layers' are accessed separately
5. Test with sample data
</implementation>

<output>
Modify files identified during research - likely:
- `./src/engines/ml_engine.py` - Model building code
- Worker training script if separate
</output>

<verification>
- Run Keras binary classification training
- No ValueError about shape
- Model trains and returns metrics
- Test with different feature counts
</verification>

<success_criteria>
- Keras binary classification trains successfully
- input_shape is correctly formed as (n_features,)
- No shape-related errors
</success_criteria>
