<objective>
Fix the /models pagination buttons that show truncated callback_data as labels instead of proper button text.
</objective>

<context>
Current bug: Buttons display "models_br...el_button" and "models_br...xt_button" - the callback_data is being shown as the button label.

Root cause: Telegram's InlineKeyboardButton requires:
- `text` parameter: What user sees (the label)
- `callback_data` parameter: Internal identifier (max 64 bytes)

The buttons are likely created with callback_data but wrong/missing text parameter.

@src/bot/handlers/ - Find models pagination implementation
</context>

<research>
1. Find where /models pagination buttons are created
2. Look for InlineKeyboardButton creation code
3. Check if text parameter is set correctly
</research>

<requirements>
1. Fix button labels to show proper text ("Previous", "Next", "◀️", "▶️")
2. Keep callback_data short (under 64 bytes)
3. Ensure callback handlers match the callback_data values
</requirements>

<implementation>
1. Find the button creation code (search for "models" + "InlineKeyboardButton")
2. Ensure format: `InlineKeyboardButton(text="◀️ Previous", callback_data="models_prev")`
3. Verify callback_data matches registered callback handlers
</implementation>

<output>
Modify the file containing models pagination button creation
</output>

<verification>
- /models buttons show readable labels
- Pagination works correctly
- No truncated text visible
</verification>

<success_criteria>
- Buttons display proper labels (not callback_data)
- Pagination navigation works
- UI is clean and professional
</success_criteria>
