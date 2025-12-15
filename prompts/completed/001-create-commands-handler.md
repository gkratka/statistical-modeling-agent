<objective>
Create a `/commands` slash command handler for the Telegram bot that lists all available commands with brief descriptions.
</objective>

<context>
This is a Telegram bot for statistical modeling and ML training. Users need a way to discover available commands.

Existing commands to list:
- /start - Initialize bot session
- /help - Get help and usage info
- /train - Start ML model training workflow
- /predict - Make predictions with trained models
- /models - View available/trained models
- /cancel - Cancel current operation
- /restart - Reset session (new)
- /commands - List all commands (this one)

@src/bot/telegram_bot.py - Main bot file where handlers are registered
@src/bot/handlers/main_handlers.py - Existing handler implementations
</context>

<requirements>
1. Create `commands_handler` async function
2. Format output as clean list with command and description
3. Register handler in telegram_bot.py
4. Follow existing handler patterns in the codebase
</requirements>

<implementation>
1. Read existing handlers to understand patterns
2. Add handler function to main_handlers.py
3. Register with `application.add_handler(CommandHandler("commands", commands_handler))`
4. Use markdown formatting for clean display
</implementation>

<output>
Modify files:
- `./src/bot/handlers/main_handlers.py` - Add commands_handler function
- `./src/bot/telegram_bot.py` - Register the handler
</output>

<verification>
- Run: `python -c "from src.bot.handlers.main_handlers import commands_handler; print('OK')"`
- Test in dev bot: Send /commands, verify list appears
</verification>

<success_criteria>
- /commands returns formatted list of all bot commands
- Each command has a brief description
- Handler follows existing code patterns
</success_criteria>
