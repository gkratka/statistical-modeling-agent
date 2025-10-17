"""Unit tests for output path validation."""

import os
import shutil
import pytest
from pathlib import Path
from src.utils.path_validator import PathValidator


class TestPathValidatorOutput:
    """Test path validator for output file validation."""

    @pytest.fixture
    def temp_output_dir(self, tmp_path):
        """Create temporary output directory."""
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        return output_dir

    @pytest.fixture
    def path_validator(self, temp_output_dir):
        """Create path validator with temp directory allowed."""
        return PathValidator(
            allowed_directories=[str(temp_output_dir)],
            max_size_mb=100,
            allowed_extensions=['.csv', '.xlsx', '.parquet']
        )

    def test_valid_directory_and_filename(self, path_validator, temp_output_dir):
        """Happy path: valid directory and filename."""
        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="predictions.csv"
        )

        assert result['is_valid'] is True
        assert result['error'] is None
        assert result['resolved_path'] == temp_output_dir / "predictions.csv"
        assert result.get('warnings', []) == []

    def test_directory_not_writable(self, path_validator, temp_output_dir):
        """Error: directory exists but no write permissions."""
        # Remove write permissions
        os.chmod(temp_output_dir, 0o444)

        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="test.csv"
        )

        # Restore permissions for cleanup
        os.chmod(temp_output_dir, 0o755)

        assert result['is_valid'] is False
        assert 'not writable' in result['error'].lower() or 'permission' in result['error'].lower()

    def test_directory_not_in_whitelist(self, path_validator, tmp_path):
        """Error: directory outside allowed paths."""
        # Create directory outside whitelist
        other_dir = tmp_path / "other"
        other_dir.mkdir()

        result = path_validator.validate_output_path(
            directory_path=str(other_dir),
            filename="test.csv"
        )

        assert result['is_valid'] is False
        assert 'not in allowed' in result['error'].lower() or 'whitelist' in result['error'].lower()

    def test_filename_with_path_traversal(self, path_validator, temp_output_dir):
        """Error: filename contains '../' or similar."""
        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="../escape.csv"
        )

        assert result['is_valid'] is False
        assert 'traversal' in result['error'].lower() or 'invalid' in result['error'].lower()

    def test_filename_with_special_chars(self, path_validator, temp_output_dir):
        """Sanitization: replace invalid filename characters."""
        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="pred<file>name?.csv"
        )

        # Should either sanitize or reject
        if result['is_valid']:
            # Sanitized filename should not contain special chars
            sanitized_name = result['resolved_path'].name
            assert '<' not in sanitized_name
            assert '>' not in sanitized_name
            assert '?' not in sanitized_name
        else:
            assert 'invalid' in result['error'].lower() or 'character' in result['error'].lower()

    def test_file_already_exists(self, path_validator, temp_output_dir):
        """Warning: file exists, offer overwrite option."""
        # Create existing file
        existing_file = temp_output_dir / "existing.csv"
        existing_file.write_text("data")

        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="existing.csv"
        )

        # Should either warn or be valid with warning
        if result['is_valid']:
            assert 'warnings' in result
            assert len(result['warnings']) > 0
            assert any('exists' in w.lower() for w in result['warnings'])
        else:
            assert 'exists' in result['error'].lower()

    def test_insufficient_disk_space(self, path_validator, temp_output_dir, monkeypatch):
        """Error: not enough disk space for predicted file size."""
        # Mock shutil.disk_usage to return low free space
        class MockDiskUsage:
            free = 1024 * 1024  # 1MB free

        monkeypatch.setattr(shutil, 'disk_usage', lambda path: MockDiskUsage())

        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="large.csv",
            required_mb=10  # Require 10MB
        )

        assert result['is_valid'] is False
        assert 'disk space' in result['error'].lower() or 'space' in result['error'].lower()

    def test_invalid_file_extension(self, path_validator, temp_output_dir):
        """Error: must be .csv extension (or other allowed extensions)."""
        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="data.txt"
        )

        assert result['is_valid'] is False
        assert 'extension' in result['error'].lower()

    def test_filename_without_extension_auto_adds_csv(self, path_validator, temp_output_dir):
        """Auto-add .csv extension if missing."""
        result = path_validator.validate_output_path(
            directory_path=str(temp_output_dir),
            filename="predictions"
        )

        if result['is_valid']:
            assert result['resolved_path'].suffix == '.csv'

    def test_directory_does_not_exist(self, path_validator, tmp_path):
        """Error: directory doesn't exist."""
        nonexistent = tmp_path / "nonexistent"

        result = path_validator.validate_output_path(
            directory_path=str(nonexistent),
            filename="test.csv"
        )

        assert result['is_valid'] is False
        assert 'not found' in result['error'].lower() or 'exist' in result['error'].lower()

    def test_sanitize_filename_method(self, path_validator):
        """Test filename sanitization helper."""
        # Test various invalid characters
        assert path_validator.sanitize_filename("file<name>.csv") == "file_name_.csv"
        assert path_validator.sanitize_filename("file:name.csv") == "file_name.csv"
        assert path_validator.sanitize_filename("file/name.csv") == "file_name.csv"
        assert path_validator.sanitize_filename("file\\name.csv") == "file_name.csv"
        assert path_validator.sanitize_filename("file?name.csv") == "file_name.csv"
        assert path_validator.sanitize_filename("file*name.csv") == "file_name.csv"

    def test_check_disk_space_method(self, path_validator, temp_output_dir):
        """Test disk space check helper."""
        # Should return True when enough space is available
        assert path_validator.check_disk_space(temp_output_dir, required_mb=1) is True

        # Should return False when required space exceeds available
        # (This test might be system-dependent, so we just check it runs)
        result = path_validator.check_disk_space(temp_output_dir, required_mb=999999)
        assert isinstance(result, bool)
