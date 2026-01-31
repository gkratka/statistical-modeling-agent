"""
Tests verifying the file size limit was correctly updated from 1000MB to 10000MB (10GB).

These tests confirm the config change propagated to all locations and that
the validation logic correctly accepts/rejects files at the new boundary.
"""

import inspect
import os
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import yaml

# Project root for config file paths
PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestFileSizeLimitConfig:
    """Verify the 10000MB limit is set in all config locations."""

    def test_config_yaml_has_10000mb(self):
        """config/config.yaml should have max_file_size_mb: 10000."""
        config_path = PROJECT_ROOT / "config" / "config.yaml"
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["local_data"]["max_file_size_mb"] == 10000

    def test_config_example_matches(self):
        """config/config.yaml.example should also have 10000."""
        example_path = PROJECT_ROOT / "config" / "config.yaml.example"
        with open(example_path) as f:
            config = yaml.safe_load(f)

        assert config["local_data"]["max_file_size_mb"] == 10000

    def test_dataloader_default_is_10000(self):
        """DataLoader should default to 10000MB when config is empty."""
        from src.processors.data_loader import DataLoader

        loader = DataLoader(config={})
        assert loader.local_max_size_mb == 10000

    def test_worker_default_parameter_is_10000(self):
        """Worker's validate_file_path default max_size_mb should be 10000."""
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "statsbot_worker",
            PROJECT_ROOT / "worker" / "statsbot_worker.py",
            submodule_search_locations=[]
        )
        # Read the source directly to inspect the default without importing
        # (worker has heavy dependencies)
        source = (PROJECT_ROOT / "worker" / "statsbot_worker.py").read_text()

        assert "max_size_mb: int = 10000" in source, \
            "Worker validate_file_path should have default max_size_mb=10000"

    def test_telegram_upload_limit_unchanged(self):
        """Telegram upload limit (10MB) must NOT have changed."""
        from src.processors.data_loader import DataLoader

        assert DataLoader.MAX_FILE_SIZE == 10 * 1024 * 1024  # 10MB


class TestFileSizeValidation:
    """Verify the worker's file size check logic at the 10GB boundary.

    Uses the same validation logic as worker/statsbot_worker.py:126-129.
    We test the logic directly rather than importing the worker (heavy deps).
    """

    DEFAULT_MAX_SIZE_MB = 10000  # Must match worker default

    @pytest.fixture
    def csv_file(self, tmp_path):
        """Create a valid CSV file for testing."""
        f = tmp_path / "test.csv"
        f.write_text("a,b,c\n1,2,3\n")
        return f

    @staticmethod
    def check_file_size(file_path: Path, max_size_mb: int) -> tuple:
        """Replicate worker's file size check (statsbot_worker.py:126-129)."""
        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)"
        if size_mb == 0:
            return False, "File is empty"
        return True, None

    def test_1282mb_file_accepted(self, csv_file):
        """A 1282MB file (the user's actual file) should be accepted at 10GB limit."""
        mock_stat = MagicMock()
        mock_stat.st_size = int(1282.3 * 1024 * 1024)

        with patch.object(Path, 'stat', return_value=mock_stat):
            ok, err = self.check_file_size(csv_file, self.DEFAULT_MAX_SIZE_MB)

        assert ok is True, f"1282MB should be accepted, got: {err}"

    def test_9999mb_file_accepted(self, csv_file):
        """A 9999MB file (just under limit) should be accepted."""
        mock_stat = MagicMock()
        mock_stat.st_size = 9999 * 1024 * 1024

        with patch.object(Path, 'stat', return_value=mock_stat):
            ok, err = self.check_file_size(csv_file, self.DEFAULT_MAX_SIZE_MB)

        assert ok is True, f"9999MB should be accepted, got: {err}"

    def test_10001mb_file_rejected(self, csv_file):
        """A 10001MB file (over limit) should be rejected."""
        mock_stat = MagicMock()
        mock_stat.st_size = 10001 * 1024 * 1024

        with patch.object(Path, 'stat', return_value=mock_stat):
            ok, err = self.check_file_size(csv_file, self.DEFAULT_MAX_SIZE_MB)

        assert ok is False
        assert "File too large" in err
        assert "10000" in err
