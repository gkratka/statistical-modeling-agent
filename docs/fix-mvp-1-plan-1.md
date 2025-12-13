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

### Fix 006: Drag-Drop File Stuck (CRITICAL)
**Prompt:** `./prompts/006-fix-drag-drop-stuck.md`

**Problem:** Bot doesn't respond when user drags/drops file during AWAITING_FILE state

**Likely Cause:** Document handler not registered or state not recognized

**Key Files:** `src/bot/telegram_bot.py`, `src/bot/handlers/`

---

### Fix 009: Back Buttons Not Working (HIGH)
**Prompt:** `./prompts/009-fix-back-buttons.md`

**Problem:** Back buttons show "at beginning of workflow" when user is mid-workflow

**Root Cause:** step_history not being populated during forward transitions

**Key Files:** `src/core/state_manager.py`, `src/bot/ml_handlers/ml_training_local_path.py`

**Reference:** /predict back buttons work - use as template

---

### Fix 004: /models Button Truncation (HIGH)
**Prompt:** `./prompts/004-fix-models-button-truncation.md`

**Problem:** Buttons show callback_data instead of labels ("models_br...el_button")

**Fix:** Ensure InlineKeyboardButton has correct `text` parameter

**Key Files:** Models handler pagination code

---

### Fix 002: Create /restart Command (HIGH)
**Prompt:** `./prompts/002-create-restart-command.md`

**Goal:** Allow users to reset session when bot gets stuck

**Implementation:** Clear session state, send confirmation

**Key Files:** `src/bot/handlers/main_handlers.py`, `src/core/state_manager.py`

---

### Fix 005: Upload Button Missing (HIGH)
**Prompt:** `./prompts/005-fix-upload-button-missing.md`

**Problem:** After selecting "Upload File", no buttons shown - user can't go back

**Fix:** Add Back/Cancel buttons and improve instructions

**Key Files:** `src/bot/ml_handlers/`

---

### Fix 007: Save Template Button (MEDIUM)
**Prompt:** `./prompts/007-fix-save-template-button.md`

**Problem:** "Save as Template" button shows error on Keras config screen

**Likely Cause:** Handler not registered or template storage not implemented

**Key Files:** `src/bot/ml_handlers/`, template storage

---

### Fix 003: /models Grouping (MEDIUM)
**Prompt:** `./prompts/003-fix-models-grouping.md`

**Goal:** Group models by category (Regression/Classification/Neural Networks) like /train

**Current:** Flat paginated list

**Key Files:** Models handler, model catalog

---

### Fix 001: Create /commands (LOW)
**Prompt:** `./prompts/001-create-commands-handler.md`

**Goal:** List all available slash commands with descriptions

**Key Files:** `src/bot/handlers/main_handlers.py`, `src/bot/telegram_bot.py`

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
- [ ] Fix 006: Drag-drop file stuck
- [ ] Fix 009: Back buttons
- [ ] Fix 004: Button truncation
- [ ] Fix 002: /restart command
- [ ] Fix 005: Upload button
- [ ] Fix 007: Save template
- [ ] Fix 003: /models grouping
- [ ] Fix 001: /commands
- [ ] All tests passing
- [ ] Manual testing complete
- [ ] PR created and merged

---

## Notes

- Images referenced in original notes describe specific UI states and error messages
- Use /predict workflow as reference for working back button implementation
- Telegram callback_data limit is 64 bytes
- Test framework: pytest
