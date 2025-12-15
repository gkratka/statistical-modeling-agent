<objective>
Create a `/restart` command that resets the user's session state when the bot gets stuck.
</objective>

<context>
Users sometimes get stuck in workflows (e.g., local file path workflow after password entry) with no way to escape except using /start. The bot needs a dedicated session reset command.

This is different from /cancel which may have workflow-specific cleanup logic.

@src/bot/telegram_bot.py - Handler registration
@src/bot/handlers/main_handlers.py - Handler implementations
@src/core/state_manager.py - Session state management
</context>

<requirements>
1. Create `/restart` command handler
2. Clear ALL user session state from StateManager
3. Reset any pending workflows
4. Send confirmation message to user
5. Must work regardless of current state
</requirements>

<implementation>
1. Check if StateManager has `reset_session()` or similar method
2. If not, add method to clear user's session completely
3. Create restart_handler in main_handlers.py:
   - Get user_id from update
   - Call state_manager.reset_session(user_id)
   - Send "Session reset. Use /start to begin again." message
4. Register handler in telegram_bot.py
</implementation>

<output>
Modify files:
- `./src/core/state_manager.py` - Add reset_session() method if needed
- `./src/bot/handlers/main_handlers.py` - Add restart_handler
- `./src/bot/telegram_bot.py` - Register handler
</output>

<verification>
- Test: Get bot into stuck state, use /restart, verify session clears
- Verify user can start fresh workflow after /restart
</verification>

<success_criteria>
- /restart clears user session state completely
- User receives confirmation message
- User can immediately start new workflows
- Works from any bot state
</success_criteria>
