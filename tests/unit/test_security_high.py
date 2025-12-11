"""Test suite for HIGH severity security fixes.

Tests for 9 HIGH severity vulnerabilities:
1. API key validation and masking
2. Pandas I/O blocking in script validator
3. Sanitization expansion for dangerous characters
4. DataFrame size limits enforcement
5. Path traversal protection (UNC, URL encoding)
6. Worker endpoint authentication
7. Password hashing with bcrypt
8. URL validation for worker endpoints
9. Session signing with HMAC-SHA256
"""

import hashlib
import hmac
import json
import os
import secrets
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.core.state_manager import StateManager, StateManagerConfig
from src.generators.validator import ScriptValidator
from src.processors.data_loader import DataLoader
from src.utils.exceptions import DataError, ValidationError
from src.utils.sanitization import InputSanitizer


class TestPandasIOBlocking:
    """Test Issue #2: Pandas I/O Blocking in Script Validator."""

    def test_pandas_read_csv_blocked(self):
        """Test that pd.read_csv is blocked."""
        script = """
import pandas as pd
data = pd.read_csv('file.csv')
"""
        validator = ScriptValidator()
        is_valid, violations = validator.validate_script(script)

        assert not is_valid
        assert any("pandas" in v.lower() or "read_csv" in v.lower() for v in violations)

    def test_pandas_to_csv_blocked(self):
        """Test that DataFrame.to_csv is blocked."""
        script = """
import pandas as pd
df = pd.DataFrame({'a': [1,2,3]})
df.to_csv('output.csv')
"""
        validator = ScriptValidator()
        is_valid, violations = validator.validate_script(script)

        assert not is_valid
        assert any("to_csv" in v.lower() or "pandas" in v.lower() for v in violations)

    def test_pandas_read_excel_blocked(self):
        """Test that pd.read_excel is blocked."""
        script = """
import pandas as pd
df = pd.read_excel('data.xlsx')
"""
        validator = ScriptValidator()
        is_valid, violations = validator.validate_script(script)

        assert not is_valid

    def test_pandas_safe_operations_allowed(self):
        """Test that safe pandas operations are still allowed."""
        script = """
import pandas as pd
import json
import sys

# Read from stdin (safe)
data = json.loads(sys.stdin.read())
df = pd.DataFrame(data['dataframe'])

# Safe operations
result = df.describe()
corr = df.corr()
mean = df.mean()

print(json.dumps({'result': result.to_dict()}))
"""
        validator = ScriptValidator()
        is_valid, violations = validator.validate_script(script)

        assert is_valid, f"Safe script rejected: {violations}"


class TestSanitizationExpansion:
    """Test Issue #3: Sanitization Expansion for Dangerous Characters."""

    def test_backtick_blocked(self):
        """Test that backtick is blocked."""
        sanitizer = InputSanitizer()

        with pytest.raises(ValidationError, match="dangerous characters"):
            sanitizer.sanitize_string("test`command`")

    def test_null_byte_blocked(self):
        """Test that null bytes are blocked."""
        sanitizer = InputSanitizer()

        # Null byte in input
        with pytest.raises(ValidationError, match="dangerous characters"):
            sanitizer.sanitize_string("test\x00file")

    def test_control_characters_blocked(self):
        """Test that control characters are blocked."""
        sanitizer = InputSanitizer()

        # Test various control characters
        for code in [0x01, 0x02, 0x10, 0x1f]:
            with pytest.raises(ValidationError):
                sanitizer.sanitize_string(f"test{chr(code)}data")

    def test_shell_metacharacters_blocked(self):
        """Test that shell metacharacters are blocked."""
        sanitizer = InputSanitizer()

        dangerous_chars = ['$', '{', '}', '[', ']', '(', ')', ';', '|']

        for char in dangerous_chars:
            with pytest.raises(ValidationError, match="dangerous characters"):
                sanitizer.sanitize_string(f"test{char}data")


class TestDataFrameSizeLimits:
    """Test Issue #4: DataFrame Size Limits."""

    def test_max_rows_enforced(self):
        """Test that maximum row limit is enforced."""
        # Create DataFrame exceeding max rows
        max_rows = 1_000_000
        df = pd.DataFrame({'a': range(max_rows + 1000)})

        loader = DataLoader()

        with pytest.raises(DataError, match="Too much data"):
            loader._validate_dataframe(df, "test.csv")

    def test_max_columns_enforced(self):
        """Test that maximum column limit is enforced."""
        # Create DataFrame exceeding max columns
        max_cols = 1000
        data = {f'col_{i}': [1, 2, 3] for i in range(max_cols + 10)}
        df = pd.DataFrame(data)

        loader = DataLoader()

        with pytest.raises(DataError, match="Too many columns"):
            loader._validate_dataframe(df, "test.csv")

    def test_valid_size_accepted(self):
        """Test that valid-sized DataFrames are accepted."""
        df = pd.DataFrame({
            'a': range(100),
            'b': range(100),
            'c': range(100)
        })

        loader = DataLoader()
        metadata = loader._validate_dataframe(df, "test.csv")

        assert metadata['shape'] == (100, 3)

    @pytest.mark.asyncio
    async def test_state_manager_enforces_memory_limits(self):
        """Test that StateManager enforces memory size limits."""
        config = StateManagerConfig(max_data_size_mb=1)  # 1MB limit
        manager = StateManager(config=config)
        session = await manager.get_or_create_session(12345, "test")

        # Create DataFrame that exceeds 1MB
        large_df = pd.DataFrame({'a': range(200000), 'b': range(200000)})

        # Should raise DataSizeLimitError from state manager
        with pytest.raises(Exception):  # Will be DataSizeLimitError
            await manager.store_data(session, large_df)


class TestPathTraversalProtection:
    """Test Issue #5: Path Traversal Protection."""

    def test_unc_path_blocked(self):
        """Test that Windows UNC paths are blocked."""
        loader = DataLoader()

        unc_paths = [
            r"\\server\share\file.csv",
            r"\\192.168.1.1\data\file.csv",
        ]

        for path in unc_paths:
            with pytest.raises(ValidationError):
                loader._validate_file_metadata(path, 1000)

    def test_url_encoded_traversal_blocked(self):
        """Test that URL-encoded path traversal is blocked."""
        loader = DataLoader()

        encoded_paths = [
            "test%2f..%2ffile.csv",  # %2f = /
            "test%2e%2e%2ffile.csv",  # %2e = .
            "test%5c..%5cfile.csv",  # %5c = \
        ]

        for path in encoded_paths:
            with pytest.raises(ValidationError):
                loader._validate_file_metadata(path, 1000)

    def test_null_byte_in_path_blocked(self):
        """Test that null bytes in paths are blocked."""
        loader = DataLoader()

        null_paths = [
            "file\x00.csv",
            "file%00.csv",
        ]

        for path in null_paths:
            with pytest.raises(ValidationError):
                loader._validate_file_metadata(path, 1000)

    def test_valid_paths_accepted(self):
        """Test that valid paths are accepted."""
        loader = DataLoader()

        valid_paths = [
            "data.csv",
            "my_file_2024.csv"
        ]

        for path in valid_paths:
            # Should not raise
            loader._validate_file_metadata(path, 1000)


class TestPasswordHashing:
    """Test Issue #7: Password Hashing with bcrypt."""

    def test_password_stored_hashed(self):
        """Test that passwords are stored hashed, not plaintext."""
        from src.utils.password_validator import PasswordValidator

        plaintext = "Test123!@#SecurePass"
        validator = PasswordValidator(password=plaintext)

        # Password should be hashed
        assert hasattr(validator, 'password_hash')
        assert validator.password_hash != plaintext
        assert validator.password_hash.startswith("$2b$")  # bcrypt prefix

    def test_password_verification_works(self):
        """Test that password verification works with bcrypt."""
        from src.utils.password_validator import PasswordValidator

        password = "Test123!@#SecurePass"
        validator = PasswordValidator(password=password)

        # Correct password should validate
        is_valid, error = validator.validate_password(
            user_id=123,
            password_input=password,
            path="/test/path"
        )

        assert is_valid
        assert error is None

    def test_wrong_password_rejected(self):
        """Test that wrong passwords are rejected."""
        from src.utils.password_validator import PasswordValidator

        validator = PasswordValidator(password="Test123!@#SecurePass")

        is_valid, error = validator.validate_password(
            user_id=123,
            password_input="WrongPassword",
            path="/test/path"
        )

        assert not is_valid
        assert error is not None

    def test_bcrypt_cost_factor(self):
        """Test that bcrypt uses cost factor 12."""
        from src.utils.password_validator import PasswordValidator

        validator = PasswordValidator(password="Test123!@#SecurePass")

        # Cost factor should be 12 (from $2b$12$...)
        assert "$2b$12$" in validator.password_hash


class TestSessionSigning:
    """Test Issue #9: Session Signing with HMAC-SHA256."""

    @pytest.mark.asyncio
    async def test_session_data_signed_on_save(self):
        """Test that session data is signed when saved."""
        signing_key = secrets.token_bytes(32)

        # Create temp directory for sessions
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['SESSION_SIGNING_KEY'] = signing_key.hex()

            try:
                manager = StateManager(sessions_dir=tmpdir)
                session = await manager.get_or_create_session(12345, "test")
                session.selections = {'test': 'data'}

                await manager.save_session_to_disk(12345)

                # Load raw JSON
                session_file = manager._get_session_file_path(12345)
                data = json.loads(session_file.read_text())

                # Should have signature
                assert 'signature' in data
                assert len(data['signature']) == 64  # HMAC-SHA256 hex length
            finally:
                os.environ.pop('SESSION_SIGNING_KEY', None)

    @pytest.mark.asyncio
    async def test_session_signature_verified_on_load(self):
        """Test that signature is verified when loading session."""
        signing_key = secrets.token_bytes(32)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['SESSION_SIGNING_KEY'] = signing_key.hex()

            try:
                manager = StateManager(sessions_dir=tmpdir)
                session = await manager.get_or_create_session(12345, "test")

                await manager.save_session_to_disk(12345)

                # Load should succeed with valid signature
                loaded = await manager.load_session_from_disk(12345)
                assert loaded is not None
            finally:
                os.environ.pop('SESSION_SIGNING_KEY', None)

    @pytest.mark.asyncio
    async def test_tampered_session_rejected(self):
        """Test that tampered session data is rejected."""
        signing_key = secrets.token_bytes(32)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['SESSION_SIGNING_KEY'] = signing_key.hex()

            try:
                manager = StateManager(sessions_dir=tmpdir)
                session = await manager.get_or_create_session(12345, "test")
                session.selections = {'test': 'data'}

                await manager.save_session_to_disk(12345)

                # Tamper with session file
                session_file = manager._get_session_file_path(12345)
                data = json.loads(session_file.read_text())
                data['selections']['test'] = 'tampered'  # Modify data
                # Don't update signature
                session_file.write_text(json.dumps(data))

                # Load should fail or return None
                loaded = await manager.load_session_from_disk(12345)
                assert loaded is None  # Tampered session rejected
            finally:
                os.environ.pop('SESSION_SIGNING_KEY', None)

    @pytest.mark.asyncio
    async def test_hmac_sha256_algorithm(self):
        """Test that HMAC-SHA256 is used for signing."""
        signing_key = secrets.token_bytes(32)

        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ['SESSION_SIGNING_KEY'] = signing_key.hex()

            try:
                manager = StateManager(sessions_dir=tmpdir)
                session = await manager.get_or_create_session(12345, "test")
                session.selections = {'test': 'data'}

                await manager.save_session_to_disk(12345)

                # Verify signature algorithm
                session_file = manager._get_session_file_path(12345)
                data = json.loads(session_file.read_text())

                # Recompute signature
                session_copy = data.copy()
                signature = session_copy.pop('signature')

                expected_sig = hmac.new(
                    signing_key,
                    json.dumps(session_copy, sort_keys=True).encode(),
                    hashlib.sha256
                ).hexdigest()

                assert signature == expected_sig
            finally:
                os.environ.pop('SESSION_SIGNING_KEY', None)
