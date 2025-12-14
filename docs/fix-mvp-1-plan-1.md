# MVP-1 Bot Fixes - Master Plan

**Branch:** `feature/fix-mvp-1`
**Total Fixes:** 9
**Prompts:** `./prompts/001-009`

---

## CRITICAL INSTRUCTION

**DO NOT STOP until all 9 fixes are completed.**

Execute fixes one at a time in the recommended order. After each fix:
1. Run relevant tests
2. Verify fix works in dev bot
3. Commit changes
4. Move to next fix

---

## Execution Order (Priority-Based)

| Order | Prompt | Fix | Priority | Complexity |
|-------|--------|-----|----------|------------|
| 1 | 008 | Keras training error | CRITICAL | Medium |
| 2 | 006 | Drag-drop file stuck | CRITICAL | Medium |
| 3 | 009 | Back buttons not working | HIGH | Medium |
| 4 | 004 | /models button truncation | HIGH | Simple |
| 5 | 002 | Create /restart command | HIGH | Medium |
| 6 | 005 | Upload button missing | HIGH | Medium |
| 7 | 007 | Save template button | MEDIUM | Medium |
| 8 | 003 | /models grouping | MEDIUM | Medium |
| 9 | 001 | Create /commands | LOW | Simple |

---

## Fix Summaries

### Fix 008: Keras Training Error (CRITICAL) ✅ COMPLETE
**Prompt:** `./prompts/completed/008-fix-keras-training-error.md`

**Error:** `ValueError: Cannot convert '(20, 'layers')' to a shape`

**Problem:** Two issues in `worker/statsbot_worker.py`:
1. Worker expected `architecture` as list but bot sent dict with `layers` key
2. Keras outputs probabilities, but `accuracy_score` needs class labels

**Fix Applied:**
- Parse architecture dict to extract layer units
- Convert Keras probability output to class labels (threshold > 0.5)

**Key Files:** `worker/statsbot_worker.py` (lines 622-746)

---

### Fix 006: Drag-Drop File Stuck (CRITICAL) ✅ COMPLETE
**Prompt:** `./prompts/completed/006-fix-drag-drop-stuck.md`

**Problem:** Bot doesn't respond when user drags/drops file during AWAITING_FILE state

**Root Cause:** Handler registration conflict - general document_handler blocked all uploads

**Fix Applied:**
- Added state filtering to general handler to skip blocking when workflow expects file upload
- Moved prediction handler to group=1 to run after ML training handlers

**Key Files:** `src/bot/main_handlers.py`, `src/bot/ml_handlers/prediction_handlers.py`

---

### Fix 009: Back Buttons Not Working (HIGH) ✅ COMPLETE
**Prompt:** `./prompts/completed/009-fix-back-buttons.md`

**Problem:** Back buttons show "at beginning of workflow" when user is mid-workflow

**Root Cause:** State history not cleared when starting new workflows - pollution from previous workflows

**Fix Applied:**
- Added `session.clear_history()` in `start_workflow()` and `cancel_workflow()` methods
- Added history clear in workflow starters in ml_training_local_path.py

**Key Files:** `src/core/state_manager.py`, `src/bot/ml_handlers/ml_training_local_path.py`

---

### Fix 004: /models Button Truncation (HIGH) ✅ COMPLETE
**Prompt:** `./prompts/completed/004-fix-models-button-truncation.md`

**Problem:** Buttons show callback_data instead of labels ("models_br...el_button")

**Root Cause:** YAML indentation error in translation files - models_browser section not properly indented under language key

**Fix Applied:**
- Fixed indentation in locales/en.yaml and locales/pt.yaml
- models_browser section now properly nested under language keys

**Key Files:** `locales/en.yaml`, `locales/pt.yaml`

---

### Fix 002: Create /restart Command (HIGH) ✅ COMPLETE
**Prompt:** `./prompts/completed/002-create-restart-command.md`

**Goal:** Allow users to reset session when bot gets stuck

**Fix Applied:**
- Added `reset_session()` method to StateManager
- Created `restart_handler` in main_handlers.py
- Clears all session data (workflow, data, history, auth state)

**Key Files:** `src/bot/main_handlers.py`, `src/core/state_manager.py`

---

### Fix 005: Upload Button Missing (HIGH) ✅ COMPLETE
**Prompt:** `./prompts/completed/005-fix-upload-button-missing.md`

**Problem:** After selecting "Upload File", no buttons shown - user can't go back

**Fix Applied:**
- Added "⬅️ Go Back" and "❌ Cancel Workflow" buttons
- Improved upload instructions with clear guidance
- Implemented callback handlers for navigation

**Key Files:** `src/bot/ml_handlers/ml_training_local_path.py`, `locales/en.yaml`

---

### Fix 007: Save Template Button (MEDIUM) ✅ COMPLETE
**Prompt:** `./prompts/completed/007-fix-save-template-button.md`

**Problem:** "Save as Template" button shows error on Keras config screen

**Root Cause:** State not set to COLLECTING_HYPERPARAMETERS before showing config complete screen

**Fix Applied:**
- Ensure session is in COLLECTING_HYPERPARAMETERS state before displaying config complete screen
- Enables valid transition to SAVING_TEMPLATE state

**Key Files:** `src/bot/ml_handlers/ml_training_local_path.py`

---

### Fix 003: /models Grouping (MEDIUM) ✅ COMPLETE
**Prompt:** `./prompts/completed/003-fix-models-grouping.md`

**Goal:** Group models by category (Regression/Classification/Neural Networks) like /train

**Fix Applied:**
- Added VIEWING_CATEGORY state to ModelsBrowserState
- Created category selection screen with 3 buttons
- Implemented category-based model filtering
- Added back to categories navigation

**Key Files:** `src/bot/ml_handlers/models_browser_handler.py`, `src/core/state_manager.py`

---

### Fix 001: Create /commands (LOW) ✅ COMPLETE
**Prompt:** `./prompts/completed/001-create-commands-handler.md`

**Goal:** List all available slash commands with descriptions

**Fix Applied:**
- Created `commands_handler` with categorized command list
- Registered handler in telegram_bot.py
- Groups: Core, ML Workflows, Utilities, Settings, Diagnostics

**Key Files:** `src/bot/main_handlers.py`, `src/bot/telegram_bot.py`

---

## Execution Instructions

To execute each fix, use:

```bash
/run-prompt [NUMBER]
```

**Recommended sequence:**
```bash
/run-prompt 008  # Keras error
/run-prompt 006  # Drag-drop
/run-prompt 009  # Back buttons
/run-prompt 004  # Button truncation
/run-prompt 002  # /restart
/run-prompt 005  # Upload button
/run-prompt 007  # Save template
/run-prompt 003  # Models grouping
/run-prompt 001  # /commands
```

---

## Testing Protocol

After each fix:

1. **Unit Tests:** `pytest tests/ --ignore=tests/unit/test_data_loader.py --ignore=tests/integration/test_data_loader_telegram.py -v`

2. **Manual Test:** Test specific functionality in dev bot

3. **Commit:** `git add . && git commit -m "fix: [description]"`

---

## Completion Checklist

- [x] Fix 008: Keras training error ✅
- [x] Fix 006: Drag-drop file stuck ✅
- [x] Fix 009: Back buttons ✅
- [x] Fix 004: Button truncation ✅
- [x] Fix 002: /restart command ✅
- [x] Fix 005: Upload button ✅
- [x] Fix 007: Save template ✅
- [x] Fix 003: /models grouping ✅
- [x] Fix 001: /commands ✅
- [ ] All tests passing
- [ ] Manual testing complete
- [ ] PR created and merged

---

## Notes

- Images referenced in original notes describe specific UI states and error messages
- Use /predict workflow as reference for working back button implementation
- Telegram callback_data limit is 64 bytes
- Test framework: pytest
