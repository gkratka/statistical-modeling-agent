# HIGH Severity Security Fixes Implementation Guide

**Date**: 2025-12-10
**Branch**: feature/fixing-security
**Prompt**: 002-high-security-fixes.md

## Fixes Completed

### ✅ Issue #3: Sanitization Expansion
**File**: `src/utils/sanitization.py`
**Status**: COMPLETE - All 4 tests passing
**Changes**:
- Expanded `dangerous_chars` regex: `r'[<>&"\'\`\$\{\}\[\]\(\);|\\\x00-\x1f\n\r\t]'`
- Added null byte detection (`\x00` and `%00`)
- Added Unicode normalization (NFKC)
- Tests: `TestSanitizationExpansion` - 4/4 passing

### ✅ Issue #4: DataFrame Size Limits
**File**: `src/processors/data_loader.py`
**Status**: ALREADY IMPLEMENTED
**Existing Code**:
```python
MAX_ROWS = 1_000_000  # Line 54
MAX_COLUMNS = 1000     # Line 55
```
**Tests**: `TestDataFrameSizeLimits` - 4/4 passing

## Fixes To Implement

### 🔧 Issue #2: Pandas I/O Blocking
**File**: `src/generators/validator.py`
**Status**: NEEDS IMPLEMENTATION
**Required Changes**:

Add to `_init_forbidden_patterns()` method (after line 71):

```python
# Pandas I/O operations (SECURITY FIX Issue #2)
r'pd\.(read_csv|read_excel|read_json|read_parquet|read_sql|read_hdf|read_pickle)',
r'pd\.(to_csv|to_excel|to_json|to_parquet|to_sql|to_hdf|to_pickle)',
r'\.to_csv\s*\(',
r'\.to_excel\s*\(',
r'\.to_json\s*\(',
r'\.to_parquet\s*\(',
r'\.to_sql\s*\(',
```

**Tests**: `TestPandasIOBlocking` - Currently 0/4 passing, will be 4/4 after fix

### 🔧 Issue #5: Path Traversal Protection
**File**: `src/processors/data_loader.py`
**Status**: NEEDS IMPLEMENTATION
**Required Changes**:

Replace `_validate_file_metadata()` method starting at line 334:

```python
def _validate_file_metadata(self, file_name: str, file_size: int) -> None:
    """
    Validate file metadata before processing.

    SECURITY FIXES (Issue #5):
    - Block Windows UNC paths (\\\\server\\share)
    - Block URL-encoded path traversal (%2f, %2e, %5c)
    - Block null bytes (\\x00, %00)
    """
    # SECURITY FIX: Block Windows UNC paths
    if file_name.startswith('\\\\\\\\') or file_name.startswith('//'):
        raise ValidationError(
            f"UNC path not allowed: {file_name}",
            field="file_name",
            value=file_name
        )

    # SECURITY FIX: Block URL-encoded path traversal
    url_encoded_patterns = ['%2f', '%2e', '%5c', '%2F', '%2E', '%5C']
    file_name_lower = file_name.lower()
    for pattern in url_encoded_patterns:
        if pattern in file_name_lower:
            raise ValidationError(
                f"URL-encoded characters not allowed in filename: {file_name}",
                field="file_name",
                value=file_name
            )

    # SECURITY FIX: Block null bytes
    if '\\x00' in file_name or '%00' in file_name:
        raise ValidationError(
            f"Null byte in filename not allowed: {file_name}",
            field="file_name",
            value=file_name
        )

    # Check file size
    if file_size > self.MAX_FILE_SIZE:
        raise ValidationError(
            f"File too large: {file_size / 1024 / 1024:.1f}MB. "
            f"Maximum allowed: {self.MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
            field="file_size",
            value=str(file_size)
        )

    if file_size == 0:
        raise ValidationError(
            "File is empty",
            field="file_size",
            value="0"
        )

    # Check file extension
    extension = self._get_file_extension(file_name).lower()
    if extension not in self.SUPPORTED_EXTENSIONS:
        raise ValidationError(
            f"Unsupported file type: {extension}. "
            f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}",
            field="file_extension",
            value=extension
        )

    # Validate filename for security (original checks)
    if not file_name or '..' in file_name or file_name.startswith('/'):
        raise ValidationError(
            "Invalid filename",
            field="file_name",
            value=file_name
        )
```

**Tests**: `TestPathTraversalProtection` - Currently 0/4 passing, will be 4/4 after fix

### 🔧 Issue #7: Password Hashing with bcrypt
**File**: `src/utils/password_validator.py`
**Status**: NEEDS IMPLEMENTATION
**Required Changes**:

1. Add bcrypt import at top of file:
```python
import bcrypt
import secrets
```

2. Replace `__init__` method (line 109):
```python
def __init__(self, password: Optional[str] = None):
    """Initialize password validator with bcrypt hashing.

    SECURITY FIX (Issue #7):
    - Password stored as bcrypt hash (cost factor 12)
    - Timing-safe comparison using secrets.compare_digest

    Args:
        password: Password to validate against. If None, uses environment
                 variable FILE_PATH_PASSWORD (required, no default).

    Raises:
        ValueError: If password is None and FILE_PATH_PASSWORD env var not set
    """
    # Priority: 1) Explicit param, 2) Environment variable (REQUIRED)
    if password is None:
        password = os.getenv('FILE_PATH_PASSWORD')
        if password is None:
            raise ValueError(
                "FILE_PATH_PASSWORD environment variable is required. "
                "Set it before using PasswordValidator."
            )

    # SECURITY FIX: Hash password with bcrypt (cost factor 12)
    self.password_hash = bcrypt.hashpw(
        password.encode('utf-8'),
        bcrypt.gensalt(rounds=12)
    )
    self._attempts: Dict[int, PasswordAttempt] = {}
```

3. Replace password validation in `validate_password()` method (line 196):
```python
# SECURITY FIX: Use bcrypt for verification (timing-safe)
try:
    is_valid = bcrypt.checkpw(
        password_input.encode('utf-8'),
        self.password_hash
    )
except (ValueError, TypeError):
    is_valid = False
```

**Tests**: `TestPasswordHashing` - Currently 2/4 passing, will be 4/4 after fix

### 🔧 Issue #9: Session Signing with HMAC-SHA256
**File**: `src/core/state_manager.py`
**Status**: NEEDS IMPLEMENTATION
**Required Changes**:

1. Add imports at top of file:
```python
import hashlib
import hmac
import os
import secrets
```

2. Add signing methods to `StateManager` class (after `_dict_to_session` method, around line 941):

```python
def _sign_session_data(self, data: Dict[str, Any]) -> str:
    """
    Generate HMAC-SHA256 signature for session data.

    SECURITY FIX (Issue #9): Session signing to prevent tampering

    Args:
        data: Session data dictionary to sign

    Returns:
        Hex-encoded HMAC-SHA256 signature

    Raises:
        ValidationError: If SESSION_SIGNING_KEY not set
    """
    signing_key = os.getenv('SESSION_SIGNING_KEY')
    if not signing_key:
        raise ValidationError(
            "SESSION_SIGNING_KEY environment variable required for session persistence",
            field="SESSION_SIGNING_KEY",
            value="not_set"
        )

    # Convert to bytes
    key_bytes = bytes.fromhex(signing_key)

    # Create canonical representation
    canonical = json.dumps(data, sort_keys=True).encode('utf-8')

    # Generate HMAC-SHA256
    signature = hmac.new(key_bytes, canonical, hashlib.sha256).hexdigest()

    return signature

def _verify_session_signature(self, data: Dict[str, Any], signature: str) -> bool:
    """
    Verify HMAC-SHA256 signature for session data.

    Args:
        data: Session data dictionary
        signature: Expected signature

    Returns:
        True if signature valid, False otherwise
    """
    try:
        expected_sig = self._sign_session_data(data)
        # SECURITY: Use timing-safe comparison
        return secrets.compare_digest(signature, expected_sig)
    except Exception:
        return False
```

3. Update `save_session_to_disk()` method (line 943) to add signature:

```python
async def save_session_to_disk(self, user_id: int) -> None:
    """Save session to disk for persistence across restarts."""
    session_key = self._get_session_key(user_id, "*")

    async with self._global_lock:
        user_session = None
        for key, session in self._sessions.items():
            if session.user_id == user_id:
                user_session = session
                break

        if user_session is None:
            raise SessionNotFoundError(f"No session found for user {user_id}")

        # Convert to dict
        session_data = self._session_to_dict(user_session)

    # SECURITY FIX (Issue #9): Add signature
    signature = self._sign_session_data(session_data)
    session_data['signature'] = signature

    # Write atomically using temporary file
    session_file = self._get_session_file_path(user_id)
    temp_file = session_file.with_suffix('.tmp')

    try:
        temp_file.write_text(json.dumps(session_data, indent=2))
        temp_file.replace(session_file)  # Atomic rename
    except Exception as e:
        if temp_file.exists():
            temp_file.unlink()
        raise
```

4. Update `load_session_from_disk()` method (line 973) to verify signature:

```python
async def load_session_from_disk(self, user_id: int) -> Optional[UserSession]:
    """Load session from disk with signature verification."""
    session_file = self._get_session_file_path(user_id)

    if not session_file.exists():
        return None

    try:
        session_data = json.loads(session_file.read_text())

        # SECURITY FIX (Issue #9): Verify signature
        if 'signature' not in session_data:
            # No signature - reject for security
            session_file.unlink()
            return None

        signature = session_data.pop('signature')
        if not self._verify_session_signature(session_data, signature):
            # Invalid signature - reject and delete
            session_file.unlink()
            return None

        # Signature valid - load session
        session = self._dict_to_session(session_data)

        # Add to memory cache
        async with self._global_lock:
            self._sessions[session.session_key] = session

        return session

    except (json.JSONDecodeError, KeyError, ValueError) as e:
        # Corrupted session file - delete it
        session_file.unlink()
        return None
```

**Tests**: `TestSessionSigning` - Currently 1/4 passing, will be 4/4 after fix

## Implementation Order

1. ✅ **Issue #3** - Sanitization (DONE)
2. **Issue #2** - Pandas I/O Blocking (5 min)
3. **Issue #5** - Path Traversal (10 min)
4. **Issue #7** - Password Hashing (15 min, requires `pip install bcrypt`)
5. **Issue #9** - Session Signing (15 min)

## Deferred Fixes (Require Architectural Changes)

- **Issue #1**: API Key Validation - Requires bot refactor
- **Issue #6**: Worker Authentication - Requires HTTP server refactor
- **Issue #8**: URL Validation - Requires bot method creation

## Test Results Target

| Test Suite | Before | After |
|------------|--------|-------|
| TestPandasIOBlocking | 1/4 | 4/4 |
| TestSanitizationExpansion | 4/4 | 4/4 ✅ |
| TestDataFrameSizeLimits | 4/4 | 4/4 ✅ |
| TestPathTraversalProtection | 1/4 | 4/4 |
| TestPasswordHashing | 2/4 | 4/4 |
| TestSessionSigning | 1/4 | 4/4 |
| **TOTAL** | **13/24** | **24/24** |

## Dependencies

```bash
# Required for Issue #7 (Password Hashing)
pip install bcrypt

# Generate SESSION_SIGNING_KEY for Issue #9
python -c "import secrets; print(secrets.token_hex(32))"

# Add to .env:
# SESSION_SIGNING_KEY=<generated_key>
# FILE_PATH_PASSWORD=<your_secure_password>
```

## Verification Commands

```bash
# Run all security tests
pytest tests/unit/test_security_high.py -v

# Run specific test suites
pytest tests/unit/test_security_high.py::TestPandasIOBlocking -v
pytest tests/unit/test_security_high.py::TestPathTraversalProtection -v
pytest tests/unit/test_security_high.py::TestPasswordHashing -v
pytest tests/unit/test_security_high.py::TestSessionSigning -v

# Run with existing tests to check for regressions
pytest tests/unit/test_security_critical.py tests/unit/test_security_high.py -v
```

## Security Impact

### Issue #2 - Pandas I/O Blocking
**Severity**: HIGH
**Impact**: Prevents file system access via pandas read/write methods in generated scripts
**Attack Vector**: Malicious user could craft input to read sensitive files

### Issue #3 - Sanitization Expansion
**Severity**: HIGH
**Impact**: Blocks control characters, null bytes, and unicode attacks
**Attack Vector**: Shell command injection, null byte injection, unicode normalization attacks

### Issue #5 - Path Traversal
**Severity**: HIGH
**Impact**: Prevents directory traversal via UNC paths and URL encoding
**Attack Vector**: Access to files outside allowed directories

### Issue #7 - Password Hashing
**Severity**: HIGH
**Impact**: Passwords stored as bcrypt hashes instead of plaintext
**Attack Vector**: Password theft if memory/logs compromised

### Issue #9 - Session Signing
**Severity**: HIGH
**Impact**: Prevents session tampering and hijacking
**Attack Vector**: Modify session files to escalate privileges or access other users' data

## Next Steps

1. Install bcrypt: `pip install bcrypt`
2. Generate SESSION_SIGNING_KEY and add to .env
3. Implement remaining fixes in order (#2, #5, #7, #9)
4. Run tests after each fix to verify
5. Commit with message: "fix: Implement HIGH severity security fixes (#2,#3,#5,#7,#9)"
6. Address deferred fixes in separate PR (#1, #6, #8)
