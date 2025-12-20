"""Tests for auto-start configuration generation.

Tests Task 6.0 - Auto-start Support for StatsBot Local Worker.
Tests generation of platform-specific auto-start configurations:
- Mac: launchd plist
- Linux: systemd user service
- Windows: Task Scheduler XML
"""

import json
import platform
import tempfile
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import Mock, patch

import pytest


# ============================================================================
# Test Data
# ============================================================================


@pytest.fixture
def worker_config() -> Dict[str, str]:
    """Sample worker configuration."""
    return {
        "ws_url": "wss://statsbot.example.com/ws",
        "token": "test-token-12345",
        "machine_id": "test-machine",
    }


@pytest.fixture
def worker_script_path() -> str:
    """Sample worker script path."""
    return "/Users/test/statsbot_worker.py"


@pytest.fixture
def temp_config_dir(tmp_path) -> Path:
    """Temporary config directory."""
    config_dir = tmp_path / ".statsbot"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


# ============================================================================
# Mac launchd Tests
# ============================================================================


class TestMacAutoStart:
    """Test Mac launchd plist generation and installation."""

    def test_generate_launchd_plist_structure(self, worker_script_path: str, worker_config: Dict):
        """Test launchd plist XML structure is valid."""
        from worker.autostart import generate_launchd_plist

        plist_content = generate_launchd_plist(worker_script_path, worker_config)

        # Verify XML structure
        assert '<?xml version="1.0" encoding="UTF-8"?>' in plist_content
        assert '<!DOCTYPE plist PUBLIC' in plist_content
        assert '<plist version="1.0">' in plist_content
        assert "<dict>" in plist_content
        assert "</plist>" in plist_content

    def test_launchd_plist_label(self, worker_script_path: str, worker_config: Dict):
        """Test launchd plist contains correct label."""
        from worker.autostart import generate_launchd_plist

        plist_content = generate_launchd_plist(worker_script_path, worker_config)

        # Verify label
        assert "<key>Label</key>" in plist_content
        assert "<string>com.statsbot.worker</string>" in plist_content

    def test_launchd_plist_program_arguments(self, worker_script_path: str, worker_config: Dict):
        """Test launchd plist contains correct program arguments."""
        from worker.autostart import generate_launchd_plist

        plist_content = generate_launchd_plist(worker_script_path, worker_config)

        # Verify program arguments
        assert "<key>ProgramArguments</key>" in plist_content
        assert "<array>" in plist_content
        assert "/usr/bin/python3" in plist_content or "python3" in plist_content
        assert worker_script_path in plist_content
        assert f"--token={worker_config['token']}" in plist_content
        assert f"--ws-url={worker_config['ws_url']}" in plist_content

    def test_launchd_plist_run_at_load(self, worker_script_path: str, worker_config: Dict):
        """Test launchd plist has RunAtLoad enabled."""
        from worker.autostart import generate_launchd_plist

        plist_content = generate_launchd_plist(worker_script_path, worker_config)

        # Verify RunAtLoad
        assert "<key>RunAtLoad</key>" in plist_content
        assert "<true/>" in plist_content

    def test_launchd_plist_keep_alive(self, worker_script_path: str, worker_config: Dict):
        """Test launchd plist has KeepAlive enabled."""
        from worker.autostart import generate_launchd_plist

        plist_content = generate_launchd_plist(worker_script_path, worker_config)

        # Verify KeepAlive
        assert "<key>KeepAlive</key>" in plist_content

    def test_install_launchd_plist(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test launchd plist installation to correct location."""
        from worker.autostart import install_launchd_plist

        with patch("pathlib.Path.home", return_value=tmp_path):
            result, message = install_launchd_plist(worker_script_path, worker_config)

            assert result is True
            assert "installed" in message.lower()

            # Verify file created
            launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
            plist_file = launch_agents_dir / "com.statsbot.worker.plist"
            assert plist_file.exists()
            assert plist_file.is_file()

            # Verify content
            content = plist_file.read_text()
            assert "com.statsbot.worker" in content
            assert worker_script_path in content

    def test_remove_launchd_plist(self, tmp_path: Path):
        """Test launchd plist removal."""
        from worker.autostart import remove_launchd_plist

        # Create dummy plist
        launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
        launch_agents_dir.mkdir(parents=True, exist_ok=True)
        plist_file = launch_agents_dir / "com.statsbot.worker.plist"
        plist_file.write_text("dummy content")

        with patch("pathlib.Path.home", return_value=tmp_path):
            result, message = remove_launchd_plist()

            assert result is True
            assert "removed" in message.lower()
            assert not plist_file.exists()

    def test_remove_launchd_plist_not_found(self, tmp_path: Path):
        """Test launchd plist removal when file doesn't exist."""
        from worker.autostart import remove_launchd_plist

        with patch("pathlib.Path.home", return_value=tmp_path):
            result, message = remove_launchd_plist()

            # Should succeed even if file not found
            assert result is True
            assert "not found" in message.lower() or "removed" in message.lower()


# ============================================================================
# Linux systemd Tests
# ============================================================================


class TestLinuxAutoStart:
    """Test Linux systemd user service generation and installation."""

    def test_generate_systemd_service_structure(self, worker_script_path: str, worker_config: Dict):
        """Test systemd service file structure is valid."""
        from worker.autostart import generate_systemd_service

        service_content = generate_systemd_service(worker_script_path, worker_config)

        # Verify systemd structure
        assert "[Unit]" in service_content
        assert "[Service]" in service_content
        assert "[Install]" in service_content

    def test_systemd_service_description(self, worker_script_path: str, worker_config: Dict):
        """Test systemd service has correct description."""
        from worker.autostart import generate_systemd_service

        service_content = generate_systemd_service(worker_script_path, worker_config)

        # Verify description
        assert "Description=" in service_content
        assert "StatsBot" in service_content

    def test_systemd_service_exec_start(self, worker_script_path: str, worker_config: Dict):
        """Test systemd service has correct ExecStart command."""
        from worker.autostart import generate_systemd_service

        service_content = generate_systemd_service(worker_script_path, worker_config)

        # Verify ExecStart
        assert "ExecStart=" in service_content
        assert "python3" in service_content
        assert worker_script_path in service_content
        assert f"--token={worker_config['token']}" in service_content
        assert f"--ws-url={worker_config['ws_url']}" in service_content

    def test_systemd_service_restart_policy(self, worker_script_path: str, worker_config: Dict):
        """Test systemd service has restart policy."""
        from worker.autostart import generate_systemd_service

        service_content = generate_systemd_service(worker_script_path, worker_config)

        # Verify restart policy
        assert "Restart=" in service_content
        assert "RestartSec=" in service_content

    def test_systemd_service_wanted_by(self, worker_script_path: str, worker_config: Dict):
        """Test systemd service has WantedBy directive."""
        from worker.autostart import generate_systemd_service

        service_content = generate_systemd_service(worker_script_path, worker_config)

        # Verify WantedBy
        assert "WantedBy=" in service_content
        assert "default.target" in service_content

    def test_install_systemd_service(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test systemd service installation to correct location."""
        from worker.autostart import install_systemd_service

        with patch("pathlib.Path.home", return_value=tmp_path):
            result, message = install_systemd_service(worker_script_path, worker_config)

            assert result is True
            assert "installed" in message.lower()

            # Verify file created
            systemd_dir = tmp_path / ".config" / "systemd" / "user"
            service_file = systemd_dir / "statsbot-worker.service"
            assert service_file.exists()
            assert service_file.is_file()

            # Verify content
            content = service_file.read_text()
            assert "[Unit]" in content
            assert worker_script_path in content

    def test_remove_systemd_service(self, tmp_path: Path):
        """Test systemd service removal."""
        from worker.autostart import remove_systemd_service

        # Create dummy service
        systemd_dir = tmp_path / ".config" / "systemd" / "user"
        systemd_dir.mkdir(parents=True, exist_ok=True)
        service_file = systemd_dir / "statsbot-worker.service"
        service_file.write_text("dummy content")

        with patch("pathlib.Path.home", return_value=tmp_path):
            result, message = remove_systemd_service()

            assert result is True
            assert "removed" in message.lower()
            assert not service_file.exists()

    def test_remove_systemd_service_not_found(self, tmp_path: Path):
        """Test systemd service removal when file doesn't exist."""
        from worker.autostart import remove_systemd_service

        with patch("pathlib.Path.home", return_value=tmp_path):
            result, message = remove_systemd_service()

            # Should succeed even if file not found
            assert result is True
            assert "not found" in message.lower() or "removed" in message.lower()


# ============================================================================
# Windows Task Scheduler Tests
# ============================================================================


class TestWindowsAutoStart:
    """Test Windows Task Scheduler task generation and installation."""

    def test_generate_task_scheduler_xml_structure(self, worker_script_path: str, worker_config: Dict):
        """Test Task Scheduler XML structure is valid."""
        from worker.autostart import generate_task_scheduler_xml

        xml_content = generate_task_scheduler_xml(worker_script_path, worker_config)

        # Verify XML structure
        assert '<?xml version="1.0" encoding="UTF-16"?>' in xml_content
        assert "<Task" in xml_content
        assert "</Task>" in xml_content
        assert "<RegistrationInfo>" in xml_content
        assert "<Triggers>" in xml_content
        assert "<Actions" in xml_content  # May have Context="Author" attribute

    def test_task_scheduler_xml_registration_info(self, worker_script_path: str, worker_config: Dict):
        """Test Task Scheduler XML has correct registration info."""
        from worker.autostart import generate_task_scheduler_xml

        xml_content = generate_task_scheduler_xml(worker_script_path, worker_config)

        # Verify registration info
        assert "<Description>" in xml_content
        assert "StatsBot" in xml_content

    def test_task_scheduler_xml_logon_trigger(self, worker_script_path: str, worker_config: Dict):
        """Test Task Scheduler XML has logon trigger."""
        from worker.autostart import generate_task_scheduler_xml

        xml_content = generate_task_scheduler_xml(worker_script_path, worker_config)

        # Verify logon trigger
        assert "<LogonTrigger>" in xml_content
        assert "<Enabled>true</Enabled>" in xml_content

    def test_task_scheduler_xml_exec_action(self, worker_script_path: str, worker_config: Dict):
        """Test Task Scheduler XML has correct exec action."""
        from worker.autostart import generate_task_scheduler_xml

        xml_content = generate_task_scheduler_xml(worker_script_path, worker_config)

        # Verify exec action
        assert "<Exec>" in xml_content
        assert "<Command>" in xml_content
        assert "python" in xml_content.lower()
        assert "<Arguments>" in xml_content
        assert worker_config["token"] in xml_content

    def test_task_scheduler_xml_settings(self, worker_script_path: str, worker_config: Dict):
        """Test Task Scheduler XML has correct settings."""
        from worker.autostart import generate_task_scheduler_xml

        xml_content = generate_task_scheduler_xml(worker_script_path, worker_config)

        # Verify settings
        assert "<Settings>" in xml_content
        assert "<StartWhenAvailable>" in xml_content
        assert "<RestartOnFailure>" in xml_content

    def test_install_task_scheduler_task_success(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test Task Scheduler task installation."""
        from worker.autostart import install_task_scheduler_task

        # Mock subprocess to simulate successful schtasks
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="SUCCESS", stderr="")

            result, message = install_task_scheduler_task(worker_script_path, worker_config)

            assert result is True
            assert "installed" in message.lower()

            # Verify schtasks was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "schtasks" in call_args
            assert "/Create" in call_args
            assert "StatsBot-Worker" in str(call_args)

    def test_install_task_scheduler_task_failure(self, worker_script_path: str, worker_config: Dict):
        """Test Task Scheduler task installation failure handling."""
        from worker.autostart import install_task_scheduler_task

        # Mock subprocess to simulate failed schtasks
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="ERROR: Access denied")

            result, message = install_task_scheduler_task(worker_script_path, worker_config)

            assert result is False
            assert "failed" in message.lower() or "error" in message.lower()

    def test_remove_task_scheduler_task_success(self):
        """Test Task Scheduler task removal."""
        from worker.autostart import remove_task_scheduler_task

        # Mock subprocess to simulate successful schtasks delete
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=0, stdout="SUCCESS", stderr="")

            result, message = remove_task_scheduler_task()

            assert result is True
            assert "removed" in message.lower()

            # Verify schtasks was called
            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert "schtasks" in call_args
            assert "/Delete" in call_args
            assert "StatsBot-Worker" in str(call_args)

    def test_remove_task_scheduler_task_not_found(self):
        """Test Task Scheduler task removal when task doesn't exist."""
        from worker.autostart import remove_task_scheduler_task

        # Mock subprocess to simulate task not found
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(returncode=1, stdout="", stderr="ERROR: The system cannot find the file specified")

            result, message = remove_task_scheduler_task()

            # Should succeed even if task not found
            assert result is True
            assert "not found" in message.lower() or "removed" in message.lower()


# ============================================================================
# Platform Detection and Auto-start Manager Tests
# ============================================================================


class TestAutoStartManager:
    """Test auto-start manager with platform detection."""

    def test_detect_platform_mac(self):
        """Test platform detection for Mac."""
        from worker.autostart import detect_platform

        with patch("platform.system", return_value="Darwin"):
            assert detect_platform() == "mac"

    def test_detect_platform_linux(self):
        """Test platform detection for Linux."""
        from worker.autostart import detect_platform

        with patch("platform.system", return_value="Linux"):
            assert detect_platform() == "linux"

    def test_detect_platform_windows(self):
        """Test platform detection for Windows."""
        from worker.autostart import detect_platform

        with patch("platform.system", return_value="Windows"):
            assert detect_platform() == "windows"

    def test_detect_platform_unsupported(self):
        """Test platform detection for unsupported OS."""
        from worker.autostart import detect_platform

        with patch("platform.system", return_value="FreeBSD"):
            assert detect_platform() == "unsupported"

    def test_install_autostart_mac(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test auto-start installation on Mac."""
        from worker.autostart import install_autostart

        with patch("platform.system", return_value="Darwin"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result, message = install_autostart(worker_script_path, worker_config)

                assert result is True
                assert "mac" in message.lower() or "launchd" in message.lower()

                # Verify plist created
                plist_file = tmp_path / "Library" / "LaunchAgents" / "com.statsbot.worker.plist"
                assert plist_file.exists()

    def test_install_autostart_linux(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test auto-start installation on Linux."""
        from worker.autostart import install_autostart

        with patch("platform.system", return_value="Linux"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result, message = install_autostart(worker_script_path, worker_config)

                assert result is True
                assert "linux" in message.lower() or "systemd" in message.lower()

                # Verify service created
                service_file = tmp_path / ".config" / "systemd" / "user" / "statsbot-worker.service"
                assert service_file.exists()

    def test_install_autostart_windows(self, worker_script_path: str, worker_config: Dict):
        """Test auto-start installation on Windows."""
        from worker.autostart import install_autostart

        with patch("platform.system", return_value="Windows"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="SUCCESS", stderr="")

                result, message = install_autostart(worker_script_path, worker_config)

                assert result is True
                assert "windows" in message.lower() or "task scheduler" in message.lower()

    def test_install_autostart_unsupported_platform(self, worker_script_path: str, worker_config: Dict):
        """Test auto-start installation on unsupported platform."""
        from worker.autostart import install_autostart

        with patch("platform.system", return_value="FreeBSD"):
            result, message = install_autostart(worker_script_path, worker_config)

            assert result is False
            assert "unsupported" in message.lower()

    def test_remove_autostart_mac(self, tmp_path: Path):
        """Test auto-start removal on Mac."""
        from worker.autostart import remove_autostart

        # Create dummy plist
        launch_agents_dir = tmp_path / "Library" / "LaunchAgents"
        launch_agents_dir.mkdir(parents=True, exist_ok=True)
        plist_file = launch_agents_dir / "com.statsbot.worker.plist"
        plist_file.write_text("dummy")

        with patch("platform.system", return_value="Darwin"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result, message = remove_autostart()

                assert result is True
                assert not plist_file.exists()

    def test_remove_autostart_linux(self, tmp_path: Path):
        """Test auto-start removal on Linux."""
        from worker.autostart import remove_autostart

        # Create dummy service
        systemd_dir = tmp_path / ".config" / "systemd" / "user"
        systemd_dir.mkdir(parents=True, exist_ok=True)
        service_file = systemd_dir / "statsbot-worker.service"
        service_file.write_text("dummy")

        with patch("platform.system", return_value="Linux"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                result, message = remove_autostart()

                assert result is True
                assert not service_file.exists()

    def test_remove_autostart_windows(self):
        """Test auto-start removal on Windows."""
        from worker.autostart import remove_autostart

        with patch("platform.system", return_value="Windows"):
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(returncode=0, stdout="SUCCESS", stderr="")

                result, message = remove_autostart()

                assert result is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestAutoStartIntegration:
    """Integration tests for complete auto-start workflow."""

    def test_install_and_remove_workflow_mac(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test complete install and remove workflow on Mac."""
        from worker.autostart import install_autostart, remove_autostart

        with patch("platform.system", return_value="Darwin"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                # Install
                result, message = install_autostart(worker_script_path, worker_config)
                assert result is True

                plist_file = tmp_path / "Library" / "LaunchAgents" / "com.statsbot.worker.plist"
                assert plist_file.exists()

                # Remove
                result, message = remove_autostart()
                assert result is True
                assert not plist_file.exists()

    def test_install_and_remove_workflow_linux(self, worker_script_path: str, worker_config: Dict, tmp_path: Path):
        """Test complete install and remove workflow on Linux."""
        from worker.autostart import install_autostart, remove_autostart

        with patch("platform.system", return_value="Linux"):
            with patch("pathlib.Path.home", return_value=tmp_path):
                # Install
                result, message = install_autostart(worker_script_path, worker_config)
                assert result is True

                service_file = tmp_path / ".config" / "systemd" / "user" / "statsbot-worker.service"
                assert service_file.exists()

                # Remove
                result, message = remove_autostart()
                assert result is True
                assert not service_file.exists()

    def test_config_persistence_for_autostart(self, worker_script_path: str, worker_config: Dict, temp_config_dir: Path):
        """Test that worker config is loaded for auto-start."""
        from worker.autostart import get_worker_config_for_autostart

        # Save config
        config_file = temp_config_dir / "config.json"
        config_file.write_text(json.dumps(worker_config))

        with patch("pathlib.Path.home", return_value=temp_config_dir.parent):
            loaded_config = get_worker_config_for_autostart()

            assert loaded_config is not None
            assert loaded_config["ws_url"] == worker_config["ws_url"]
            assert loaded_config["token"] == worker_config["token"]

    def test_config_not_found_for_autostart(self, temp_config_dir: Path):
        """Test handling when config not found for auto-start."""
        from worker.autostart import get_worker_config_for_autostart

        # Use non-existent directory
        with patch("pathlib.Path.home", return_value=temp_config_dir / "nonexistent"):
            loaded_config = get_worker_config_for_autostart()

            assert loaded_config is None
