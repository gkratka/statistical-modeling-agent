<objective>
Fix all non-working back buttons in the Telegram bot's /train and /predict workflows.
Some back buttons work correctly (e.g., after model selection in /predict), but others fail or don't navigate properly.
Remove back buttons where they don't make functional sense (e.g., immediately after /train command).
</objective>

<context>
This is a Telegram bot for ML model training and predictions.
The bot uses inline keyboard buttons for navigation, including "Back" buttons.
Back buttons should restore the previous state and re-display the previous menu/prompt.

Key workflow files to examine:
@src/bot/ml_handlers/
@src/bot/handlers/
@src/core/state_manager.py

The state_manager.py handles conversation state and likely has state history functionality.
</context>

<research>
Before making changes, thoroughly investigate:

1. **Working back buttons**: Find examples where back buttons work correctly (e.g., /predict after model selection)
   - How do they handle state restoration?
   - What callback_data pattern do they use?
   - How do they re-render the previous menu?

2. **Broken back buttons**: Identify ALL instances where back buttons don't work
   - List each workflow step with a broken back button
   - Determine WHY each is broken (missing handler? wrong state? no history?)

3. **Unnecessary back buttons**: Find places where back button doesn't make sense
   - Entry points (immediately after /train, /predict commands)
   - Points of no return (after training starts, after prediction submits)

4. **State management**: Examine how state history works
   - Does state_manager store history for back navigation?
   - How is previous state retrieved and restored?
</research>

<requirements>
1. **Fix all broken back buttons** in /train and /predict workflows
   - Ensure clicking "Back" restores previous state
   - Ensure clicking "Back" re-displays the correct previous menu/message
   - Use consistent pattern matching working back buttons

2. **Remove unnecessary back buttons** where navigation back doesn't make sense:
   - Immediately after /train command (can't go back before start)
   - Immediately after /predict command
   - After irreversible actions (training started, prediction submitted)

3. **Maintain consistency** - all back buttons should behave the same way

4. **Test each fix** - verify the back button navigates correctly
</requirements>

<implementation>
Follow the pattern used by WORKING back buttons in the codebase.

For fixing broken back buttons:
1. Ensure state is saved before transitioning forward
2. Ensure back handler retrieves previous state correctly
3. Ensure back handler re-renders the correct previous menu

For removing unnecessary back buttons:
1. Simply remove the back button from the inline keyboard
2. Don't remove functionality, just the button where it doesn't apply
</implementation>

<output>
Modify files as needed in:
- `./src/bot/ml_handlers/` - ML workflow handlers
- `./src/bot/handlers/` - General handlers
- `./src/core/state_manager.py` - If state history needs fixes

Document each change:
- Which file was modified
- What back button was fixed/removed
- Why the change was needed
</output>

<verification>
Before declaring complete, verify:
1. List all back buttons in /train workflow - each should work or be removed with justification
2. List all back buttons in /predict workflow - each should work or be removed with justification
3. No broken back buttons remain
4. Removed back buttons make sense (entry points, irreversible actions only)
</verification>

<success_criteria>
- All back buttons in /train and /predict workflows either work correctly OR are removed
- Working back buttons restore previous state and display previous menu
- Removed back buttons are only at entry points or after irreversible actions
- Consistent behavior across all workflows
</success_criteria>
</content>
</invoke>