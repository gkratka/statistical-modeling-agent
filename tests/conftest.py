"""
Pytest fixtures for path validator tests.

Author: Statistical Modeling Agent
Created: 2025-10-06
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_test_env():
    """Create temporary directory structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)

        # Create allowed directory structure
        allowed_dir = tmppath / "allowed_data"
        allowed_dir.mkdir()

        # Create test files
        (allowed_dir / "valid_file.csv").write_text("col1,col2\n1,2\n3,4\n")
        (allowed_dir / "large_file.csv").write_text("data\n" * 10000)
        (allowed_dir / "empty_file.csv").write_text("")
        (allowed_dir / "wrong_ext.txt").write_text("col1,col2\n1,2\n")

        # Create subdirectory
        subdir = allowed_dir / "subdir"
        subdir.mkdir()
        (subdir / "nested_file.csv").write_text("a,b\n1,2\n")

        # Create restricted directory (outside whitelist)
        restricted_dir = tmppath / "restricted"
        restricted_dir.mkdir()
        (restricted_dir / "restricted_file.csv").write_text("secret,data\n")

        yield {
            "tmpdir": tmppath,
            "allowed_dir": allowed_dir,
            "restricted_dir": restricted_dir,
            "valid_file": allowed_dir / "valid_file.csv",
            "large_file": allowed_dir / "large_file.csv",
            "empty_file": allowed_dir / "empty_file.csv",
            "wrong_ext": allowed_dir / "wrong_ext.txt",
            "nested_file": subdir / "nested_file.csv",
            "restricted_file": restricted_dir / "restricted_file.csv"
        }
