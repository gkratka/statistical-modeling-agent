<objective>
Fix the "Save as Template" button that shows "Error Occurred" when clicked on the Keras Configuration Complete screen.
</objective>

<context>
Bug: On Keras Configuration Complete screen, clicking "Save as Template" button shows generic "Error Occurred" message.

Possible causes:
1. Callback handler not registered for save_template action
2. Template storage logic not implemented
3. Handler throws exception
4. Missing required data in session state

@src/bot/ml_handlers/ - ML workflow handlers
</context>

<research>
1. Find the callback handler for save_template (search for "save_template" or "template")
2. Check if handler is registered in telegram_bot.py
3. Look for template storage implementation
4. Check error logs for specific exception
</research>

<requirements>
1. Save as Template button must work without errors
2. Template should persist user's Keras configuration
3. User should see confirmation message on success
4. If templates not fully implemented, show "Coming soon" instead of error
</requirements>

<implementation>
1. Find the save_template callback handler
2. If handler missing, create it:
   ```python
   async def handle_save_template(update, context):
       query = update.callback_query
       await query.answer()
       # Get current configuration from session
       # Save to template storage
       await query.edit_message_text("✅ Template saved successfully!")
   ```
3. If template storage not implemented, either:
   - Implement basic template storage (JSON file per user)
   - Or show "Feature coming soon" message
4. Register handler if not registered
</implementation>

<output>
Modify files:
- `./src/bot/ml_handlers/` - Template handler
- `./src/bot/telegram_bot.py` - Register handler if needed
- Template storage logic (create if needed)
</output>

<verification>
- Click "Save as Template" → no error
- Confirmation message appears
- If implemented: template can be loaded later
</verification>

<success_criteria>
- Button click doesn't cause error
- User gets clear feedback
- If fully implemented: templates persist and can be reused
</success_criteria>
