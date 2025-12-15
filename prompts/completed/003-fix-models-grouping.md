<objective>
Restructure the `/models` command to group models by category (like /train workflow) instead of showing a flat paginated list.
</objective>

<context>
Current behavior: /models shows flat list with pagination (Page 1/4):
- Linear Regression, Ridge Regression, Lasso Regression, ElasticNet...

Desired behavior: Match /train workflow grouping:
- Regression Models → [Linear, Ridge, Lasso, ElasticNet, etc.]
- Classification Models → [Logistic, Decision Tree, Random Forest, etc.]
- Neural Networks → [Keras models]

User should first select a category, then see models in that category.

@src/bot/handlers/ - Find models handler
@src/bot/ml_handlers/ - ML-related handlers
</context>

<research>
1. Find where /models handler is implemented
2. Find where /train model selection is implemented (for reference)
3. Identify model catalog/config defining available models
</research>

<requirements>
1. Show category buttons first: "Regression", "Classification", "Neural Networks"
2. On category selection, show models in that category
3. Include "Back" button to return to categories
4. Maintain any existing model info display functionality
</requirements>

<implementation>
1. Locate models handler and understand current implementation
2. Create category selection screen with inline keyboard
3. Create callback handlers for each category
4. Reuse model grouping logic from /train if available
5. Add navigation (back to categories)
</implementation>

<output>
Modify files identified during research - likely:
- `./src/bot/handlers/` - Models handler
- Callback handlers for category selection
</output>

<verification>
- /models shows 3 category buttons
- Clicking category shows relevant models
- Back button returns to categories
</verification>

<success_criteria>
- Models grouped by category matching /train structure
- Clean navigation between categories and model lists
- Consistent UX with rest of bot
</success_criteria>
