"""
Comprehensive security tests for local file path validation.

This test suite validates all security layers of the path validation system
to ensure no vulnerabilities exist that could lead to unauthorized file access.

Test Coverage:
- Path traversal detection (15+ attack scenarios)
- Directory whitelist enforcement
- File extension validation
- File size limits
- Permission handling
- Edge cases (symlinks, special chars, encoding)

Author: Statistical Modeling Agent
Created: 2025-10-06 (Phase 1: Security Foundation)
"""

import os
import tempfile
from pathlib import Path

import pytest

from src.utils.path_validator import (
    validate_local_path,
    is_path_in_allowed_directory,
    detect_path_traversal,
    get_file_size_mb,
    sanitize_path_for_display,
    validate_allowed_directories
)
from src.utils.exceptions import PathValidationError


class TestPathValidation:
    """Test comprehensive path validation with security focus."""

    def test_valid_path_in_whitelist(self, temp_test_env):
        """Valid path in allowed directory should pass all checks."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["valid_file"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv", ".xlsx"]
        )

        assert is_valid is True
        assert error_msg is None
        assert resolved is not None
        assert resolved.exists()

    def test_nested_file_in_allowed_subdirectory(self, temp_test_env):
        """Files in subdirectories of allowed dirs should be valid."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["nested_file"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is True
        assert error_msg is None

    def test_path_traversal_parent_blocked(self, temp_test_env):
        """Path traversal using ../ should be blocked."""
        attack_path = str(temp_test_env["allowed_dir"] / ".." / "restricted" / "restricted_file.csv")

        is_valid, error_msg, resolved = validate_local_path(
            attack_path,
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "traversal" in error_msg.lower() or "not in allowed" in error_msg.lower()

    def test_path_traversal_multiple_parents_blocked(self, temp_test_env):
        """Multiple ../ path traversal should be blocked."""
        attack_path = str(temp_test_env["allowed_dir"] / ".." / ".." / "etc" / "passwd")

        is_valid, error_msg, resolved = validate_local_path(
            attack_path,
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv", ""]  # Allow no extension for /etc/passwd
        )

        assert is_valid is False

    def test_path_outside_whitelist_blocked(self, temp_test_env):
        """Paths outside allowed directories should be rejected."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["restricted_file"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "not in allowed" in error_msg.lower()

    def test_file_not_found_rejected(self, temp_test_env):
        """Non-existent files should be rejected."""
        nonexistent = temp_test_env["allowed_dir"] / "does_not_exist.csv"

        is_valid, error_msg, resolved = validate_local_path(
            str(nonexistent),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "not found" in error_msg.lower()

    def test_oversized_file_rejected(self, temp_test_env):
        """Files exceeding size limit should be rejected."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["large_file"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=0.001,  # Tiny limit (1KB)
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "too large" in error_msg.lower()

    def test_invalid_extension_rejected(self, temp_test_env):
        """Invalid file extensions should be rejected."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["wrong_ext"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv", ".xlsx"]  # .txt not allowed
        )

        assert is_valid is False
        assert "invalid" in error_msg.lower() or "extension" in error_msg.lower()

    def test_directory_rejected(self, temp_test_env):
        """Directories should be rejected (files only)."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["allowed_dir"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "not a file" in error_msg.lower() or "directory" in error_msg.lower()

    def test_zero_byte_file_rejected(self, temp_test_env):
        """Empty (0 byte) files should be rejected."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["empty_file"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "empty" in error_msg.lower()

    def test_case_insensitive_extension(self, temp_test_env):
        """File extensions should be case-insensitive."""
        csv_upper = temp_test_env["allowed_dir"] / "test.CSV"
        csv_upper.write_text("data\n")

        is_valid, error_msg, resolved = validate_local_path(
            str(csv_upper),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]  # lowercase
        )

        assert is_valid is True

    def test_relative_path_resolved(self, temp_test_env):
        """Relative paths should be resolved to absolute."""
        # Save current directory
        original_cwd = os.getcwd()

        try:
            # Change to allowed directory
            os.chdir(temp_test_env["allowed_dir"])

            # Use relative path
            is_valid, error_msg, resolved = validate_local_path(
                "./valid_file.csv",  # Relative path
                [str(temp_test_env["allowed_dir"])],
                max_size_mb=100,
                allowed_extensions=[".csv"]
            )

            assert is_valid is True
            assert resolved is not None
            assert resolved.is_absolute()
        finally:
            os.chdir(original_cwd)

    @pytest.mark.skipif(os.name == 'nt', reason="Unix symlinks not supported on Windows")
    def test_symlink_within_whitelist_allowed(self, temp_test_env):
        """Symlinks pointing within whitelist should be allowed."""
        symlink = temp_test_env["allowed_dir"] / "link_to_valid.csv"
        symlink.symlink_to(temp_test_env["valid_file"])

        is_valid, error_msg, resolved = validate_local_path(
            str(symlink),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is True

    @pytest.mark.skipif(os.name == 'nt', reason="Unix symlinks not supported on Windows")
    def test_symlink_outside_whitelist_blocked(self, temp_test_env):
        """Symlinks pointing outside whitelist should be blocked."""
        symlink = temp_test_env["allowed_dir"] / "link_to_restricted.csv"
        symlink.symlink_to(temp_test_env["restricted_file"])

        is_valid, error_msg, resolved = validate_local_path(
            str(symlink),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is False
        assert "not in allowed" in error_msg.lower()


class TestPathTraversalDetection:
    """Test path traversal attack detection."""

    def test_unix_path_traversal_detected(self):
        """Unix-style ../ traversal should be detected."""
        assert detect_path_traversal("../../etc/passwd") is True
        assert detect_path_traversal("/data/../../../etc/passwd") is True

    def test_windows_path_traversal_detected(self):
        """Windows-style ..\\ traversal should be detected."""
        assert detect_path_traversal("..\\..\\windows\\system32") is True
        assert detect_path_traversal("C:\\data\\..\\..\\windows") is True

    def test_url_encoded_traversal_detected(self):
        """URL-encoded path traversal should be detected."""
        assert detect_path_traversal("%2e%2e%2fetc%2fpasswd") is True
        assert detect_path_traversal("..%2f..%2fetc") is True

    def test_mixed_encoding_traversal_detected(self):
        """Mixed encoding traversal should be detected."""
        assert detect_path_traversal("..%2f%2e%2e%5c") is True

    def test_clean_path_not_detected(self):
        """Clean paths should not trigger detection."""
        assert detect_path_traversal("/data/users/file.csv") is False
        assert detect_path_traversal("C:\\Users\\data\\file.csv") is False
        assert detect_path_traversal("./data/file.csv") is False  # relative but not traversal

    def test_case_insensitive_detection(self):
        """Detection should be case-insensitive."""
        assert detect_path_traversal("..\\WINDOWS\\System32") is True
        assert detect_path_traversal("%2E%2E%2F") is True


class TestWhitelistEnforcement:
    """Test directory whitelist enforcement."""

    def test_path_in_single_allowed_directory(self):
        """Path directly in allowed directory should pass."""
        assert is_path_in_allowed_directory(
            Path("/data/users/file.csv"),
            ["/data/users"]
        ) is True

    def test_path_in_subdirectory_of_allowed(self):
        """Path in subdirectory of allowed directory should pass."""
        assert is_path_in_allowed_directory(
            Path("/data/users/subdir/file.csv"),
            ["/data/users"]
        ) is True

    def test_path_outside_all_allowed_directories(self):
        """Path outside all allowed directories should fail."""
        assert is_path_in_allowed_directory(
            Path("/etc/passwd"),
            ["/data/users", "/home/data"]
        ) is False

    def test_path_in_multiple_allowed_directories(self):
        """Path matching any allowed directory should pass."""
        # Use paths that actually exist or are resolvable
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            dir1 = tmppath / "dir1"
            dir2 = tmppath / "dir2"
            dir3 = tmppath / "dir3"
            dir1.mkdir()
            dir2.mkdir()
            dir3.mkdir()

            test_file = dir2 / "file.csv"
            test_file.write_text("data")

            assert is_path_in_allowed_directory(
                test_file,
                [str(dir1), str(dir2), str(dir3)]
            ) is True

    def test_partial_match_rejected(self):
        """Partial directory name match should be rejected."""
        # /data/users_backup should NOT match /data/users
        assert is_path_in_allowed_directory(
            Path("/data/users_backup/file.csv"),
            ["/data/users"]
        ) is False


class TestFileSizeCalculation:
    """Test file size calculation utilities."""

    def test_get_file_size_mb_small_file(self):
        """Small files should report correct size."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("small data\n")
            f.flush()
            path = Path(f.name)

        try:
            size_mb = get_file_size_mb(path)
            assert 0 < size_mb < 0.001  # Less than 1KB
        finally:
            path.unlink()

    def test_get_file_size_mb_medium_file(self):
        """Medium files should report correct size."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            # Write ~1MB of data
            f.write("x" * (1024 * 1024))
            f.flush()
            path = Path(f.name)

        try:
            size_mb = get_file_size_mb(path)
            assert 0.9 < size_mb < 1.1  # Approximately 1MB
        finally:
            path.unlink()

    def test_get_file_size_mb_nonexistent_file(self):
        """Nonexistent files should raise PathValidationError."""
        with pytest.raises(PathValidationError):
            get_file_size_mb(Path("/nonexistent/file.csv"))


class TestPathSanitization:
    """Test path sanitization for display."""

    def test_short_path_unchanged(self):
        """Short paths should be returned unchanged."""
        path = "/data/file.csv"
        sanitized = sanitize_path_for_display(path, max_length=100)
        assert sanitized == path

    def test_long_path_truncated(self):
        """Long paths should be truncated in the middle."""
        path = "/very/long/path/to/some/deeply/nested/directory/file.csv"
        sanitized = sanitize_path_for_display(path, max_length=30)
        assert len(sanitized) <= 30
        assert "..." in sanitized
        assert "file.csv" in sanitized  # Filename preserved

    def test_very_long_filename_truncated(self):
        """Very long filenames should be truncated."""
        path = "/data/" + "x" * 200 + ".csv"
        sanitized = sanitize_path_for_display(path, max_length=50)
        assert len(sanitized) <= 50
        assert "..." in sanitized


class TestAllowedDirectoriesValidation:
    """Test validation of allowed_directories configuration."""

    def test_valid_absolute_paths(self):
        """Valid absolute paths should pass."""
        is_valid, error = validate_allowed_directories([
            "/home/user/data",
            "/var/data/users"
        ])
        assert is_valid is True
        assert error is None

    def test_relative_path_rejected(self):
        """Relative paths should be rejected."""
        is_valid, error = validate_allowed_directories([
            "./data",
            "/home/user/data"
        ])
        assert is_valid is False
        assert "relative" in error.lower()

    def test_dangerous_system_directories_rejected(self):
        """System directories should be rejected."""
        dangerous_dirs = [
            ["/etc"],
            ["/bin"],
            ["/root"],
            ["/boot"],
        ]

        for dirs in dangerous_dirs:
            is_valid, error = validate_allowed_directories(dirs)
            assert is_valid is False, f"Directory {dirs} should be rejected but got is_valid={is_valid}"
            if error:
                assert "system" in error.lower() or "not allowed" in error.lower()

    def test_empty_list_rejected(self):
        """Empty allowed directories list should be rejected."""
        is_valid, error = validate_allowed_directories([])
        assert is_valid is False
        assert "no allowed" in error.lower()


class TestSecurityScenarios:
    """Integration tests for complete security scenarios."""

    def test_multiple_security_layers_all_pass(self, temp_test_env):
        """Valid file should pass all security layers."""
        is_valid, error_msg, resolved = validate_local_path(
            str(temp_test_env["valid_file"]),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv", ".xlsx"]
        )

        # All layers should pass
        assert is_valid is True
        assert error_msg is None
        assert resolved is not None
        assert resolved.exists()
        assert resolved.is_file()
        assert os.access(resolved, os.R_OK)

    def test_attack_chain_blocked_early(self, temp_test_env):
        """Attack paths should be blocked at earliest layer."""
        # Path traversal attack
        attack_path = "../../etc/passwd"

        is_valid, error_msg, resolved = validate_local_path(
            attack_path,
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[""]  # Allow any extension for this test
        )

        # Should be blocked (either traversal detection or whitelist)
        assert is_valid is False
        assert error_msg is not None

    def test_edge_case_unicode_path(self, temp_test_env):
        """Unicode characters in paths should be handled."""
        unicode_file = temp_test_env["allowed_dir"] / "文件.csv"
        unicode_file.write_text("data\n")

        is_valid, error_msg, resolved = validate_local_path(
            str(unicode_file),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is True

    def test_edge_case_spaces_in_path(self, temp_test_env):
        """Spaces in file paths should be handled."""
        space_file = temp_test_env["allowed_dir"] / "file with spaces.csv"
        space_file.write_text("data\n")

        is_valid, error_msg, resolved = validate_local_path(
            str(space_file),
            [str(temp_test_env["allowed_dir"])],
            max_size_mb=100,
            allowed_extensions=[".csv"]
        )

        assert is_valid is True
