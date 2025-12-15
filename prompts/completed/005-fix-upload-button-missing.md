<objective>
Add proper UI elements (Cancel/Back buttons and improved instructions) to the file upload screen in /train workflow.
</objective>

<context>
Current behavior: After selecting "Upload File" in data source selection, user sees only "Please upload your file" text with no buttons.

Problem: User has no way to go back or cancel. Telegram doesn't support file upload buttons directly (users must drag/drop or attach), but we should provide navigation.

@src/bot/ml_handlers/ - ML training handlers
@src/bot/ml_handlers/ml_training_local_path.py - File upload handling
</context>

<requirements>
1. Add "◀️ Back" button to return to data source selection
2. Add "❌ Cancel" button to exit workflow
3. Improve instructions explaining how to upload (attach icon or drag/drop)
4. Optionally add file format guidance (.csv, .xlsx supported)
</requirements>

<implementation>
1. Find the handler that shows "Please upload your file" message
2. Add InlineKeyboardMarkup with Back and Cancel buttons
3. Update message text with clearer instructions:
   ```
   📤 Upload Your File

   To upload, either:
   • Click the 📎 attachment icon and select your file
   • Drag and drop your file into this chat

   Supported formats: .csv, .xlsx, .xls
   ```
4. Add callback handlers for Back and Cancel buttons
</implementation>

<output>
Modify files:
- `./src/bot/ml_handlers/` - File upload handler
- Add/update callback handlers for navigation
</output>

<verification>
- Upload screen shows Back and Cancel buttons
- Back returns to data source selection
- Cancel exits workflow completely
- Instructions are clear
</verification>

<success_criteria>
- User can navigate away from upload screen
- Instructions clearly explain upload process
- Buttons work correctly
</success_criteria>
