# Language Detection Implementation for /start Handler

## Overview
Implemented automatic language detection for `/start` and `/help` commands using the `detect_and_set_language` decorator. Users can now test Portuguese translations by sending a Portuguese message before using commands.

## Implementation Details

### 1. New Decorator: `detect_and_set_language`

**Location:** `/Users/gkratka/Documents/statistical-modeling-agent/src/utils/decorators.py`

**Purpose:** Auto-detect user language from message text and update session

**Features:**
- Detects language from message text (>3 characters required)
- Updates session with detected language, confidence, and timestamp
- Uses cached language for performance (avoids re-detection)
- Gracefully handles missing state_manager

**Usage Pattern:**
```python
@telegram_handler
@detect_and_set_language  # Add this before handler
@log_user_action("Bot start")
async def start_handler(update, context):
    # Language already detected and stored in session
    session = await state_manager.get_session(...)
    locale = session.language
    await update.message.reply_text(I18nManager.t('welcome', locale=locale))
```

### 2. Updated Handlers

#### /start Handler
**File:** `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers.py`

**Changes:**
1. Added `@detect_and_set_language` decorator
2. Retrieves `locale` from session
3. Passes `locale` to `get_welcome_message()`

**Decorator Stack:**
```python
@telegram_handler           # Error handling + validation
@detect_and_set_language    # Language detection (NEW)
@log_user_action("Bot start")  # Logging
async def start_handler(update, context):
    ...
```

#### /help Handler
**File:** `/Users/gkratka/Documents/statistical-modeling-agent/src/bot/handlers.py`

**Changes:**
- Same pattern as `/start` handler
- Detects language and passes to `get_help_message()`

### 3. Detection Strategy

**Workflow:**
1. User sends message (e.g., "Olá, preciso de ajuda")
2. Decorator detects language → Portuguese (confidence: 0.95)
3. Session updated: `session.language = "pt"`
4. User sends `/start`
5. Handler retrieves `locale="pt"` from session
6. Response uses Portuguese translations (if available)

**Caching:**
- First message: Full detection (library or LLM)
- Subsequent messages: Cached result (confidence: 1.0)
- No re-detection overhead

**Short Text Handling:**
- Commands like `/start` (≤3 chars) skip detection
- Defaults to `"en"` until longer message received

## Test Coverage

### Integration Tests
**File:** `tests/integration/test_language_detection_integration.py`

**Test Cases (9 total, all passing):**

1. **test_start_english_default**
   - `/start` without prior message → English

2. **test_start_portuguese_after_detection**
   - Portuguese message → `/start` → Portuguese response

3. **test_help_command_language_detection**
   - Portuguese message → `/help` → Portuguese response

4. **test_language_persistence_across_commands**
   - Language persists across multiple commands
   - Uses cached detection

5. **test_short_text_no_detection**
   - Short text (<3 chars) doesn't trigger detection

6. **test_missing_state_manager_graceful_fallback**
   - Handler works without state_manager (defaults to English)

7. **test_detection_confidence_stored**
   - Confidence and timestamp stored in session

8. **test_decorator_updates_session**
   - Decorator correctly updates session language

9. **test_decorator_preserves_handler_metadata**
   - Decorator preserves handler metadata (functools.wraps)

**Test Execution:**
```bash
$ python3 -m pytest tests/integration/test_language_detection_integration.py -v
# Result: 9 passed in 0.91s
```

### Manual Testing
**Script:** `scripts/test_language_detection.py`

**Test Scenarios:**
1. Language detector unit tests (6 test cases)
2. Translation file verification (en.yaml, pt.yaml)
3. Manual Telegram bot testing instructions

**Execution:**
```bash
$ python3 scripts/test_language_detection.py
# Outputs: Test results + manual test instructions
```

## Usage Examples

### Example 1: English (Default)
```
User: /start
Bot:  Welcome to Statistical Modeling Agent!
      🔧 Version: DataLoader-v2.0-NUCLEAR-FIX
      ...
```

### Example 2: Portuguese Detection
```
User: Olá, preciso de ajuda com análise de dados
Bot:  [Response acknowledges message]
      [Language detected: pt, confidence: 0.95]

User: /start
Bot:  Bem-vindo ao Agente de Modelagem Estatística!
      🔧 Versão: DataLoader-v2.0-NUCLEAR-FIX
      ...
```

### Example 3: Language Persistence
```
User: Quero treinar um modelo
Bot:  [Language: pt, cached]

User: /help
Bot:  [Portuguese help message]
      [No re-detection, uses cached language]

User: Hello, I need help
Bot:  [Language switched to: en]

User: /start
Bot:  [English welcome message]
```

## Session Data

**Language Fields (UserSession):**
```python
session.language: str = "en"  # ISO 639-1 code
session.language_detected_at: Optional[datetime] = None
session.language_detection_confidence: float = 0.0  # 0-1
```

**Example Session After Detection:**
```python
{
    "user_id": 12345,
    "language": "pt",
    "language_detected_at": "2025-01-27T10:30:00",
    "language_detection_confidence": 0.95,
    ...
}
```

## Performance Characteristics

**First Message (Detection):**
- Library detection: ~10ms (langdetect)
- LLM fallback (if needed): ~500ms (Claude API)

**Subsequent Messages (Cached):**
- Cache lookup: <1ms
- No API calls
- Confidence: 1.0

**Memory Overhead:**
- 3 fields per session (~50 bytes)
- Language detector singleton (shared)

## Dependencies

**Required:**
- `langdetect` - Primary detection library
- `python-i18n` - Translation management
- Existing: `LanguageDetector`, `I18nManager`, `StateManager`

**Optional:**
- `anthropic` - LLM fallback (for ambiguous cases)

## Configuration

**LanguageDetector Settings:**
```python
detector = LanguageDetector(
    confidence_threshold=0.8,      # Minimum library confidence
    use_llm_fallback=True,         # Enable Claude fallback
    supported_languages=["en", "pt"]  # Supported codes
)
```

**StateManager Settings:**
- No configuration changes required
- Language fields auto-initialized
- Session persistence includes language data

## Backward Compatibility

**Default Behavior:**
- Users without language preference → English
- Missing state_manager → English fallback
- No translations loaded → Keys shown (graceful degradation)

**Migration:**
- Existing sessions: `language="en"` (default)
- No breaking changes to handlers
- Decorator optional (can be added incrementally)

## Future Enhancements

**Potential Improvements:**
1. **Manual language selection:** `/language pt` command
2. **Language auto-switch:** Detect language change mid-conversation
3. **More languages:** Spanish, French, German support
4. **Translation coverage:** Expand to all bot messages
5. **User preferences:** Persistent language preference in database

## Files Modified

1. **src/utils/decorators.py**
   - Added `detect_and_set_language` decorator
   - Imported `LanguageDetector`

2. **src/bot/handlers.py**
   - Updated `/start` handler with decorator
   - Updated `/help` handler with decorator
   - Added locale retrieval from session

3. **tests/integration/test_language_detection_integration.py** (NEW)
   - 9 integration tests (all passing)
   - Tests decorator, handlers, and session integration

4. **scripts/test_language_detection.py** (NEW)
   - Manual testing script
   - Test instructions for Telegram bot

## Success Criteria

✅ English works by default
✅ Portuguese detected from user message
✅ /start responds in detected language
✅ Language persists in session
✅ Dev bot testable
✅ All tests passing (9/9)
✅ Backward compatible (no breaking changes)

## Developer Notes

**Testing with Dev Bot:**
1. Start bot: `python3 src/bot/telegram_bot.py`
2. Send Portuguese message: "Olá, preciso de ajuda"
3. Check logs: "Language detected: user=..., lang=pt"
4. Send `/start` → Should see Portuguese (if pt.yaml loaded)
5. Send English message → Language switches to English
6. Send `/start` → Should see English

**Debugging:**
- Check logs for: `"Language detected: user=..., lang=..."`
- Verify session: `session.language`, `session.language_detection_confidence`
- Test with different message lengths (>3 chars for detection)

**Common Issues:**
1. **Translation keys shown instead of text:**
   - Check I18nManager initialized
   - Verify pt.yaml exists in locales/
   - Ensure YAML keys match (e.g., `commands.start.welcome`)

2. **Language not detected:**
   - Message too short (<3 chars)
   - state_manager not in bot_data
   - Check langdetect library installed

3. **Language not persisting:**
   - Session not being updated (check logs)
   - state_manager.update_session() not called
   - Session timeout (default 30 min)

## Related Documentation

- **Phase 1 Implementation:** `docs/implementation/i18n-phase1-implementation.md`
- **Language Detector:** `src/utils/language_detector.py`
- **I18n Manager:** `src/utils/i18n_manager.py`
- **State Manager:** `src/core/state_manager.py`
- **Translation Files:** `locales/en.yaml`, `locales/pt.yaml`

---

**Implementation Date:** 2025-01-27
**Status:** Complete and Tested
**Test Coverage:** 9/9 passing
