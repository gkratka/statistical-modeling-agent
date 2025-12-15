<objective>
Fix the bug where dragging/dropping a file to the chat causes the bot to get stuck and not respond.
</objective>

<context>
Bug: User dragged german_credit_data_train2.csv (64.8KB) to chat during AWAITING_FILE state, but bot didn't respond. Bot appears stuck.

Possible causes:
1. Document handler not registered properly
2. State machine doesn't recognize document uploads during AWAITING_FILE state
3. Handler filtering excludes document messages
4. Handler throws exception silently

@src/bot/handlers/ - Document handlers
@src/bot/telegram_bot.py - Handler registration
@src/core/state_manager.py - State definitions
</context>

<research>
1. Find document handler (handles MessageHandler with filters.Document)
2. Check handler registration order (filters may conflict)
3. Check if handler validates session state before processing
4. Look for error handling that might swallow exceptions
</research>

<requirements>
1. Bot must respond to document uploads during file upload state
2. Process the uploaded file through data_loader
3. Continue workflow after successful upload
4. Show clear error if file is invalid
</requirements>

<implementation>
1. Verify document handler exists and is registered
2. Check handler priority/order - document handler should run for AWAITING_FILE state
3. Add logging to track document receipt
4. Ensure handler doesn't silently fail:
   ```python
   async def handle_document(update, context):
       logger.info(f"Document received: {update.message.document.file_name}")
       # ... processing
   ```
5. Test with various file sizes and types
</implementation>

<output>
Modify files as identified during research - likely:
- `./src/bot/telegram_bot.py` - Handler registration
- `./src/bot/handlers/` - Document handler
</output>

<verification>
- Drag/drop file during upload state → bot responds
- File is processed correctly
- Workflow continues to next step
- Invalid files show error message
</verification>

<success_criteria>
- Drag/drop uploads work reliably
- Bot never gets stuck on file upload
- Clear feedback on success/failure
</success_criteria>
