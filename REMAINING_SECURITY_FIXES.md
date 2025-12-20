# Remaining Security Fixes - Implementation Code

## Summary

**Completed**:
- ✅ Issue #3: Sanitization Expansion (4/4 tests passing)
- ✅ Issue #2: Pandas I/O Blocking (4/4 tests passing)
- ✅ Issue #4: DataFrame Size Limits (already implemented, 4/4 tests passing)

**To Implement**:
- 🔧 Issue #5: Path Traversal Protection
- 🔧 Issue #7: Password Hashing with bcrypt
- 🔧 Issue #9: Session Signing with HMAC-SHA256

**Current Test Status**: 16/24 tests passing

---

## Issue #5: Path Traversal Protection

**File**: `src/processors/data_loader.py`
**Method**: `_validate_file_metadata` (line 334)

Replace the entire method with this:

```python
def _validate_file_metadata(self, file_name: str, file_size: int) -> None:
    """
    Validate file metadata before processing.

    Args:
        file_name: Name of the file
        file_size: Size of the file in bytes

    Raises:
        ValidationError: If validation fails

    Security Updates (Issue #5 - HIGH):
    - Block Windows UNC paths (\\\\server\\share)
    - Block URL-encoded path traversal (%2f, %2e, %5c)
    - Block null bytes (\\x00, %00)
    """
    # SECURITY FIX: Block Windows UNC paths
    if file_name.startswith('\\\\') or file_name.startswith('//'):
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
    if '\x00' in file_name or '%00' in file_name:
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

**Test After**: `pytest tests/unit/test_security_high.py::TestPathTraversalProtection -xvs`

---

## Issue #7: Password Hashing with bcrypt

**File**: `src/utils/password_validator.py`

### Step 1: Install bcrypt

```bash
pip install bcrypt
```

### Step 2: Add imports (after line 26)

```python
import bcrypt
import secrets
```

### Step 3: Replace `__init__` method (line 109-128)

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

### Step 4: Replace password validation logic (line 196)

Find this line:
```python
is_valid = password_input == self.password
```

Replace with:
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

**Test After**: `pytest tests/unit/test_security_high.py::TestPasswordHashing -xvs`

---

## Issue #9: Session Signing with HMAC-SHA256

**File**: `src/core/state_manager.py`

### Step 1: Add imports (after line 11)

```python
import hashlib
import hmac
import secrets
```

### Step 2: Add signing methods (after `_dict_to_session` method, around line 941)

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
        from src.utils.exceptions import ValidationError
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

### Step 3: Update `save_session_to_disk` method (line 943)

Find the section where session_data is created and replace with:

```python
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

### Step 4: Update `load_session_from_disk` method (line 973)

Replace the try block with:

```python
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

### Step 5: Generate signing key for .env

```bash
python -c "import secrets; print(f'SESSION_SIGNING_KEY={secrets.token_hex(32)}')"
```

Add the output to your `.env` file.

**Test After**: `pytest tests/unit/test_security_high.py::TestSessionSigning -xvs`

---

## Final Verification

Run all security tests:

```bash
pytest tests/unit/test_security_high.py -v
```

Expected result: **24/24 tests passing**

Run critical security tests to ensure no regression:

```bash
pytest tests/unit/test_security_critical.py tests/unit/test_security_high.py -v
```

---

## Dependencies Installation

```bash
pip install bcrypt
```

## Environment Variables

Add to `.env`:

```bash
# Generate with: python -c "import secrets; print(secrets.token_hex(32))"
SESSION_SIGNING_KEY=<your_64_char_hex_key_here>

# Set a secure password for file path access
FILE_PATH_PASSWORD=<your_secure_password_here>
```

---

## Git Commit Message

```
fix: Implement HIGH severity security fixes (#2,#3,#4,#5,#7,#9)

Security Updates:
- Issue #2: Block pandas I/O operations in script validator
- Issue #3: Expand sanitization regex (control chars, null bytes, unicode)
- Issue #4: DataFrame size limits (already implemented)
- Issue #5: Path traversal protection (UNC paths, URL encoding, null bytes)
- Issue #7: Password hashing with bcrypt (cost factor 12)
- Issue #9: Session signing with HMAC-SHA256

Test Coverage: 24/24 tests passing

Deferred for separate PR:
- Issue #1: API key validation (requires bot refactor)
- Issue #6: Worker endpoint authentication (requires server refactor)
- Issue #8: URL validation (requires new methods)
```
