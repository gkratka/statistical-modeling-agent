<objective>
Fix back buttons that show "Cannot Go Back - You're at the beginning of the workflow" even when user is mid-workflow.
</objective>

<context>
Bug: Back buttons don't work at multiple workflow stages:
- After "Access Granted" for directory → Back shows "at beginning"
- After "Schema Accepted" → Back shows "at beginning"

However, back buttons DO work in /predict workflow after model selection.

Root cause: The `step_history` in session state isn't being populated as user progresses through workflow steps.

@src/core/state_manager.py - Step history management
@src/bot/ml_handlers/ml_training_local_path.py - Local path training handlers
@src/bot/ml_handlers/ - Other ML handlers
</context>

<research>
1. Find how step_history is managed in StateManager
2. Compare working back buttons (/predict) with broken ones (/train local path)
3. Find where steps SHOULD be pushed to history but aren't
4. Check the back button handler logic
</research>

<requirements>
1. Back buttons must work at all mid-workflow screens
2. Each forward step must push to step_history
3. Back button must pop from history and restore previous state
4. "At beginning" message only when history is truly empty
</requirements>

<implementation>
1. Find the back button handler:
   ```python
   async def handle_back_button(update, context):
       session = await state_manager.get_session(user_id, conv_id)
       if not session.step_history:
           await query.answer("Cannot go back - at beginning")
           return
       previous_step = session.step_history.pop()
       # Restore previous state
   ```
2. Find all workflow step transitions that should push to history
3. Add step_history.append() at each forward transition:
   ```python
   # Before transitioning to new step
   session.step_history.append({
       'state': current_state,
       'data': relevant_data
   })
   await state_manager.update_session(session)
   ```
4. Use /predict as reference for correct implementation
</implementation>

<output>
Modify files:
- `./src/core/state_manager.py` - If step_history management needs fixes
- `./src/bot/ml_handlers/ml_training_local_path.py` - Add step history pushes
- Other handlers missing step history tracking
</output>

<verification>
- Start local path workflow
- Progress through steps
- Click Back at various points → returns to previous screen
- Back at true beginning shows appropriate message
</verification>

<success_criteria>
- All Back buttons work mid-workflow
- User can navigate backwards through completed steps
- State is properly restored when going back
- Only shows "at beginning" when actually at beginning
</success_criteria>
