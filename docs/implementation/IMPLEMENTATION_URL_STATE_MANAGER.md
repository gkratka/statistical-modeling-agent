# State Manager URL Workflow Implementation

## Overview
This document describes the State Manager updates for the online URL dataset workflow feature.

## Changes Made

### 1. New State Constants (MLTrainingState)
Added three new states for URL workflow:

```python
AWAITING_DATASET_URL = "awaiting_dataset_url"           # User provides dataset URL
VALIDATING_DATASET_URL = "validating_dataset_url"       # Bot validating URL
CONFIRMING_DETECTED_SCHEMA = "confirming_detected_schema" # User reviewing detected schema
```

### 2. New UserSession Fields

#### URL-specific fields:
```python
dataset_url: Optional[str] = None                  # Normalized URL if using URL source
dataset_source_type: Optional[str] = None          # Explicit tracking: "telegram", "local_path", or "url"
```

#### Updated existing field:
```python
data_source: Optional[str] = None                  # Now accepts: "telegram", "local_path", or "url"
```

### 3. State Transitions

#### From CHOOSING_DATA_SOURCE:
- → `AWAITING_DATASET_URL` (new: user chose URL option)
- → `AWAITING_FILE_PATH` (existing: local path)
- → `AWAITING_DATA` (existing: Telegram upload)
- → `LOADING_TEMPLATE` (existing: use template)

#### From AWAITING_DATASET_URL:
- → `VALIDATING_DATASET_URL` (user provided URL)

#### From VALIDATING_DATASET_URL:
- → `CONFIRMING_DETECTED_SCHEMA` (URL valid, schema detected)
- → `AWAITING_DATASET_URL` (URL invalid, retry)

#### From CONFIRMING_DETECTED_SCHEMA:
- → `SELECTING_TARGET` (schema accepted, legacy path)
- → `CONFIRMING_MODEL` (schema accepted, skip to model selection)
- → `AWAITING_DATASET_URL` (schema rejected, try different URL)

### 4. Session Persistence

#### Persisted fields:
- `dataset_url` - The normalized URL string
- `dataset_source_type` - Explicit source type tracking

#### Security exclusions (NOT persisted):
- `dynamic_allowed_directories` - Session-scoped only
- `pending_auth_path` - Security: requires re-authentication
- `password_attempts` - Rate limiting reset on reload

### 5. Test Coverage

Created comprehensive test suite: `tests/unit/test_state_manager_url.py`

**24 tests covering:**

1. **State Existence** (3 tests)
   - Verify all new states are defined

2. **Data Fields** (4 tests)
   - URL field existence and mutability
   - Valid source type values

3. **State Transitions** (7 tests)
   - All valid URL workflow transitions
   - Success and retry paths
   - Schema acceptance/rejection flows

4. **Invalid Transitions** (3 tests)
   - Cannot skip validation step
   - Cannot jump from unrelated states
   - Proper transition validation

5. **Session Persistence** (4 tests)
   - URL fields properly persisted
   - None values handled correctly
   - Restoration from disk works

6. **Integration Workflows** (3 tests)
   - Complete URL workflow from start to finish
   - URL validation retry workflow
   - Schema rejection workflow

**Test Results:**
- 24/24 tests passing (100% pass rate)
- 15 existing password tests still passing
- Total: 39 tests passing

## User Workflow

```
1. /train command
   ↓
2. CHOOSING_DATA_SOURCE: Choose "Online Dataset (URL)"
   ↓
3. AWAITING_DATASET_URL: User provides URL
   ↓
4. VALIDATING_DATASET_URL: Bot validates URL and loads data
   ↓ (success)
5. CONFIRMING_DETECTED_SCHEMA: Show detected schema
   ↓ (accept)
6. CONFIRMING_MODEL: Continue to model selection
   ↓
   [Existing training workflow continues...]
```

## Error Handling

### URL Validation Failure:
```
VALIDATING_DATASET_URL → AWAITING_DATASET_URL (retry)
```

### Schema Rejection:
```
CONFIRMING_DETECTED_SCHEMA → AWAITING_DATASET_URL (try different URL)
```

## Integration Points

### With URL Validator (next component):
```python
# In handler:
url_validator = URLValidator(config)
is_valid, normalized_url, error = url_validator.validate_url(user_input)

if is_valid:
    session.dataset_url = normalized_url
    session.dataset_source_type = "url"
    await state_manager.transition_state(session, VALIDATING_DATASET_URL)
else:
    # Show error, retry
    await state_manager.transition_state(session, AWAITING_DATASET_URL)
```

### With Schema Detector (existing component):
```python
# During validation:
schema = schema_detector.detect_schema(df)
session.detected_schema = schema
await state_manager.transition_state(session, CONFIRMING_DETECTED_SCHEMA)
```

## Files Modified

1. `/Users/gkratka/Documents/statistical-modeling-agent/src/core/state_manager.py`
   - Added 3 new state constants
   - Added 2 new UserSession fields
   - Added 3 state transition mappings
   - Updated persistence methods

2. `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_state_manager_url.py`
   - Created comprehensive test suite (24 tests)

## Security Considerations

1. **URL Validation**: State machine expects URL validation in `VALIDATING_DATASET_URL` state
2. **No Persistence of Sensitive Data**: `pending_auth_path` NOT persisted for security
3. **Session Isolation**: URL data fields are session-specific
4. **Proper State Flow**: Cannot skip validation step (enforced by transitions)

## Next Steps

1. Implement URL Validator (`src/utils/url_validator.py`)
2. Create URL workflow handlers (`src/bot/handlers/ml_training_url.py`)
3. Add URL-specific messages (`src/bot/messages/url_messages.py`)
4. Update data loader to support URL sources
5. Integration testing with complete workflow

## Compatibility

- Backward compatible with existing workflows (local path, Telegram upload)
- Existing tests still pass (15 password tests)
- No breaking changes to existing state machine logic
- Clean separation of URL workflow states
