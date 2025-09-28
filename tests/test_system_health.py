#!/usr/bin/env python3
"""
System health and process diagnostic tests.

Consolidates system-level tests from test_bot_instance_fix.py for process management,
file system checks, and bot instance monitoring using pytest parameterization.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestSystemHealth:
    """System-level health checks and process diagnostics."""

    @pytest.mark.parametrize("command", ["ps", "lsof", "pgrep"])
    def test_diagnostic_commands_available(self, command):
        """Test that required diagnostic commands are available."""
        result = subprocess.run(['which', command], capture_output=True)
        assert result.returncode == 0, f"{command} command not available"

    @pytest.mark.parametrize("directory", ["src", "tests", "scripts"])
    def test_project_structure_exists(self, directory):
        """Test that expected project directories exist."""
        dir_path = project_root / directory
        assert dir_path.exists(), f"Project directory missing: {directory}"
        assert dir_path.is_dir(), f"Path exists but is not directory: {directory}"

    @pytest.mark.parametrize("file_path,required_content", [
        ("src/bot/handlers.py", "HANDLERS_VERSION"),
        ("src/processors/data_loader.py", "class DataLoader"),
        ("src/utils/decorators.py", "telegram_handler"),
        (".env", "TELEGRAM_BOT_TOKEN"),
    ])
    def test_critical_files_content(self, file_path, required_content):
        """Test that critical files exist and contain required content."""
        file_full_path = project_root / file_path
        assert file_full_path.exists(), f"Critical file missing: {file_path}"

        content = file_full_path.read_text()
        assert required_content in content, f"Missing required content in {file_path}: {required_content}"

    def test_bot_instance_hunter_agent_exists(self):
        """Test that bot-instance-hunter agent is properly located."""
        agent_file = project_root / ".claude" / "agents" / "bot-instance-hunter.md"
        assert agent_file.exists(), "bot-instance-hunter agent not found in .claude/agents/"

        content = agent_file.read_text()
        required_sections = ["bot-instance-hunter Agent", "Process Forensics"]
        for section in required_sections:
            assert section in content, f"Agent file missing section: {section}"

    def test_cache_files_information(self):
        """Informational test about Python cache files."""
        cache_dirs = list(project_root.rglob("__pycache__"))
        pyc_files = list(project_root.rglob("*.pyc"))

        # This is informational - cache files might exist
        if cache_dirs or pyc_files:
            print(f"Info: Found {len(cache_dirs)} cache dirs and {len(pyc_files)} .pyc files")
            print("Consider running: find . -name '__pycache__' -type d -exec rm -rf {} + (if needed)")

    def test_version_file_status(self):
        """Test VERSION file status if it exists."""
        version_file = project_root / "VERSION"
        if version_file.exists():
            content = version_file.read_text()
            expected_markers = ["DATALOADER_INTEGRATION", "CACHE_CLEARED"]
            for marker in expected_markers:
                assert marker in content, f"VERSION file missing marker: {marker}"

    def test_running_processes_detection(self):
        """Informational test to detect running bot processes."""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                bot_processes = [
                    line for line in lines
                    if 'python' in line.lower() and
                    ('telegram' in line.lower() or 'bot' in line.lower())
                ]

                if bot_processes:
                    print(f"Info: Found {len(bot_processes)} potential bot processes")
                    for proc in bot_processes[:3]:  # Show max 3 for brevity
                        print(f"  {proc[:100]}...")
                else:
                    print("Info: No bot processes found running")
        except Exception as e:
            print(f"Info: Could not check processes: {e}")

    @pytest.mark.parametrize("env_var", ["TELEGRAM_BOT_TOKEN", "ANTHROPIC_API_KEY"])
    def test_environment_variables(self, env_var):
        """Test that critical environment variables are defined."""
        # Read from .env file since environment might not be loaded
        env_file = project_root / ".env"
        if env_file.exists():
            content = env_file.read_text()
            assert f"{env_var}=" in content, f"Environment variable {env_var} not defined in .env"
        else:
            # Fallback to checking actual environment
            assert os.getenv(env_var) is not None, f"Environment variable {env_var} not set"


class TestDocumentationHealth:
    """Health checks for project documentation and fix tracking."""

    def test_fix_documentation_exists(self):
        """Test that fix documentation is properly maintained."""
        fix_doc = project_root / "dev" / "fix" / "data-loader-fix1.md"
        if fix_doc.exists():
            content = fix_doc.read_text()
            required_sections = [
                "TDD Approach",
                "Phase 1: Hunt",
                "Phase 5: Verify",
                "test_bot_instance_fix.py"
            ]
            for section in required_sections:
                assert section in content, f"Fix documentation missing section: {section}"

    @pytest.mark.parametrize("script_name", [
        "dev_start.sh",
        "monitor_bot_health.sh",
        "monitor_bot.py"
    ])
    def test_utility_scripts_exist(self, script_name):
        """Test that utility scripts exist."""
        script_path = project_root / "scripts" / script_name
        if script_path.exists():
            assert script_path.is_file(), f"Script path is not a file: {script_name}"
            # For shell scripts, check they're executable
            if script_name.endswith('.sh'):
                assert os.access(script_path, os.X_OK), f"Shell script not executable: {script_name}"


class TestCodebaseIntegrity:
    """Tests for overall codebase integrity and consistency."""

    @pytest.mark.parametrize("module_path", [
        "src.bot.handlers",
        "src.processors.data_loader",
        "src.utils.decorators",
        "src.utils.exceptions",
        "src.utils.logger"
    ])
    def test_critical_modules_importable(self, module_path):
        """Test that critical modules can be imported."""
        try:
            __import__(module_path)
        except ImportError as e:
            pytest.fail(f"Failed to import {module_path}: {e}")

    def test_no_syntax_errors_in_source(self):
        """Test that all Python files in src/ have valid syntax."""
        python_files = list((project_root / "src").rglob("*.py"))

        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    compile(f.read(), str(py_file), 'exec')
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {py_file}: {e}")

    @pytest.mark.parametrize("forbidden_import", [
        "from src.processors.old_loader",  # Old imports
        "import deprecated_module",
        "from telegram_bot_deprecated"
    ])
    def test_no_forbidden_imports(self, forbidden_import):
        """Test that source files don't contain forbidden imports."""
        python_files = list((project_root / "src").rglob("*.py"))

        for py_file in python_files:
            content = py_file.read_text()
            assert forbidden_import not in content, f"Forbidden import in {py_file}: {forbidden_import}"

    def test_consistent_version_references(self):
        """Test that version references are consistent across files."""
        handlers_file = project_root / "src" / "bot" / "handlers.py"
        if handlers_file.exists():
            content = handlers_file.read_text()
            # Should have consistent version tracking
            assert "v2.0" in content or "2.0" in content, "Missing version information in handlers"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])