"""Auto-start configuration for StatsBot Worker.

This module provides platform-specific auto-start configuration generation:
- Mac: launchd plist in ~/Library/LaunchAgents/
- Linux: systemd user service in ~/.config/systemd/user/
- Windows: Task Scheduler task via schtasks.exe

Usage:
    from worker.autostart import install_autostart, remove_autostart

    # Install auto-start
    success, message = install_autostart(script_path, config)

    # Remove auto-start
    success, message = remove_autostart()
"""

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple


# ============================================================================
# Platform Detection
# ============================================================================


def detect_platform() -> str:
    """
    Detect current operating system platform.

    Returns:
        Platform identifier: "mac", "linux", "windows", or "unsupported"
    """
    system = platform.system()

    if system == "Darwin":
        return "mac"
    elif system == "Linux":
        return "linux"
    elif system == "Windows":
        return "windows"
    else:
        return "unsupported"


# ============================================================================
# Mac launchd Support
# ============================================================================


def generate_launchd_plist(script_path: str, config: Dict[str, str]) -> str:
    """
    Generate launchd plist XML for Mac auto-start.

    Args:
        script_path: Path to worker script
        config: Worker configuration with ws_url and token

    Returns:
        XML content for launchd plist
    """
    # Find python3 executable
    python_path = shutil.which("python3") or "/usr/bin/python3"

    plist_template = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.statsbot.worker</string>

    <key>ProgramArguments</key>
    <array>
        <string>{python_path}</string>
        <string>{script_path}</string>
        <string>--token={config['token']}</string>
        <string>--ws-url={config['ws_url']}</string>
    </array>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>StandardOutPath</key>
    <string>{str(Path.home())}/.statsbot/logs/worker.log</string>

    <key>StandardErrorPath</key>
    <string>{str(Path.home())}/.statsbot/logs/worker.error.log</string>

    <key>WorkingDirectory</key>
    <string>{str(Path.home())}/.statsbot</string>

    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
    </dict>
</dict>
</plist>
"""
    return plist_template


def install_launchd_plist(script_path: str, config: Dict[str, str]) -> Tuple[bool, str]:
    """
    Install launchd plist for Mac auto-start.

    Args:
        script_path: Path to worker script
        config: Worker configuration

    Returns:
        Tuple of (success, message)
    """
    try:
        # Generate plist content
        plist_content = generate_launchd_plist(script_path, config)

        # Create LaunchAgents directory
        launch_agents_dir = Path.home() / "Library" / "LaunchAgents"
        launch_agents_dir.mkdir(parents=True, exist_ok=True)

        # Write plist file
        plist_file = launch_agents_dir / "com.statsbot.worker.plist"
        plist_file.write_text(plist_content)

        # Set proper permissions
        plist_file.chmod(0o644)

        # Create log directory
        log_dir = Path.home() / ".statsbot" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        return True, f"Auto-start installed successfully for Mac (launchd)\nPlist location: {plist_file}"

    except Exception as e:
        return False, f"Failed to install auto-start for Mac: {str(e)}"


def remove_launchd_plist() -> Tuple[bool, str]:
    """
    Remove launchd plist for Mac auto-start.

    Returns:
        Tuple of (success, message)
    """
    try:
        plist_file = Path.home() / "Library" / "LaunchAgents" / "com.statsbot.worker.plist"

        if not plist_file.exists():
            return True, "Auto-start not found (already removed)"

        # Remove plist file
        plist_file.unlink()

        return True, "Auto-start removed successfully for Mac (launchd)"

    except Exception as e:
        return False, f"Failed to remove auto-start for Mac: {str(e)}"


# ============================================================================
# Linux systemd Support
# ============================================================================


def generate_systemd_service(script_path: str, config: Dict[str, str]) -> str:
    """
    Generate systemd user service for Linux auto-start.

    Args:
        script_path: Path to worker script
        config: Worker configuration with ws_url and token

    Returns:
        Content for systemd service file
    """
    # Find python3 executable
    python_path = shutil.which("python3") or "/usr/bin/python3"

    service_template = f"""[Unit]
Description=StatsBot Local Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={python_path} {script_path} --token={config['token']} --ws-url={config['ws_url']}
Restart=on-failure
RestartSec=10
StandardOutput=append:{str(Path.home())}/.statsbot/logs/worker.log
StandardError=append:{str(Path.home())}/.statsbot/logs/worker.error.log
WorkingDirectory={str(Path.home())}/.statsbot

[Install]
WantedBy=default.target
"""
    return service_template


def install_systemd_service(script_path: str, config: Dict[str, str]) -> Tuple[bool, str]:
    """
    Install systemd user service for Linux auto-start.

    Args:
        script_path: Path to worker script
        config: Worker configuration

    Returns:
        Tuple of (success, message)
    """
    try:
        # Generate service content
        service_content = generate_systemd_service(script_path, config)

        # Create systemd user directory
        systemd_dir = Path.home() / ".config" / "systemd" / "user"
        systemd_dir.mkdir(parents=True, exist_ok=True)

        # Write service file
        service_file = systemd_dir / "statsbot-worker.service"
        service_file.write_text(service_content)

        # Set proper permissions
        service_file.chmod(0o644)

        # Create log directory
        log_dir = Path.home() / ".statsbot" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        # Reload systemd daemon (optional - may fail if systemd not running)
        try:
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # systemctl may not be available or timeout - non-fatal
            pass

        message = f"""Auto-start installed successfully for Linux (systemd)
Service location: {service_file}

To enable and start the worker now:
  systemctl --user enable statsbot-worker.service
  systemctl --user start statsbot-worker.service

To check status:
  systemctl --user status statsbot-worker.service
"""
        return True, message

    except Exception as e:
        return False, f"Failed to install auto-start for Linux: {str(e)}"


def remove_systemd_service() -> Tuple[bool, str]:
    """
    Remove systemd user service for Linux auto-start.

    Returns:
        Tuple of (success, message)
    """
    try:
        service_file = Path.home() / ".config" / "systemd" / "user" / "statsbot-worker.service"

        if not service_file.exists():
            return True, "Auto-start not found (already removed)"

        # Stop service if running (optional - may fail)
        try:
            subprocess.run(
                ["systemctl", "--user", "stop", "statsbot-worker.service"],
                check=False,
                capture_output=True,
                timeout=5,
            )
            subprocess.run(
                ["systemctl", "--user", "disable", "statsbot-worker.service"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # systemctl may not be available - non-fatal
            pass

        # Remove service file
        service_file.unlink()

        # Reload daemon (optional)
        try:
            subprocess.run(
                ["systemctl", "--user", "daemon-reload"],
                check=False,
                capture_output=True,
                timeout=5,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return True, "Auto-start removed successfully for Linux (systemd)"

    except Exception as e:
        return False, f"Failed to remove auto-start for Linux: {str(e)}"


# ============================================================================
# Windows Task Scheduler Support
# ============================================================================


def generate_task_scheduler_xml(script_path: str, config: Dict[str, str]) -> str:
    """
    Generate Task Scheduler XML for Windows auto-start.

    Args:
        script_path: Path to worker script
        config: Worker configuration with ws_url and token

    Returns:
        XML content for Task Scheduler task
    """
    # Find python executable
    python_path = shutil.which("python") or shutil.which("python3") or "python"

    # Build arguments
    arguments = f'"{script_path}" --token={config["token"]} --ws-url={config["ws_url"]}'

    xml_template = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>StatsBot Local Worker - Automatic ML job execution</Description>
    <URI>\\StatsBot-Worker</URI>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>{python_path}</Command>
      <Arguments>{arguments}</Arguments>
      <WorkingDirectory>{str(Path.home() / '.statsbot')}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""
    return xml_template


def install_task_scheduler_task(script_path: str, config: Dict[str, str]) -> Tuple[bool, str]:
    """
    Install Task Scheduler task for Windows auto-start.

    Args:
        script_path: Path to worker script
        config: Worker configuration

    Returns:
        Tuple of (success, message)
    """
    try:
        # Generate XML content
        xml_content = generate_task_scheduler_xml(script_path, config)

        # Create temp XML file
        temp_dir = Path.home() / ".statsbot" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        xml_file = temp_dir / "statsbot-worker-task.xml"
        xml_file.write_text(xml_content, encoding="utf-16")

        # Create task using schtasks
        cmd = [
            "schtasks",
            "/Create",
            "/TN",
            "StatsBot-Worker",
            "/XML",
            str(xml_file),
            "/F",  # Force create (overwrite if exists)
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Clean up temp XML
        try:
            xml_file.unlink()
        except:
            pass

        if result.returncode == 0:
            message = """Auto-start installed successfully for Windows (Task Scheduler)
Task name: StatsBot-Worker

To view or manage the task:
  1. Open Task Scheduler (taskschd.msc)
  2. Look for "StatsBot-Worker" in the task list

To start the task now:
  schtasks /Run /TN "StatsBot-Worker"
"""
            return True, message
        else:
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, f"Failed to create Task Scheduler task: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Failed to install auto-start: schtasks command timeout"
    except FileNotFoundError:
        return False, "Failed to install auto-start: schtasks.exe not found (Windows only)"
    except Exception as e:
        return False, f"Failed to install auto-start for Windows: {str(e)}"


def remove_task_scheduler_task() -> Tuple[bool, str]:
    """
    Remove Task Scheduler task for Windows auto-start.

    Returns:
        Tuple of (success, message)
    """
    try:
        cmd = [
            "schtasks",
            "/Delete",
            "/TN",
            "StatsBot-Worker",
            "/F",  # Force delete without confirmation
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return True, "Auto-start removed successfully for Windows (Task Scheduler)"
        else:
            # Check if error is because task doesn't exist
            error_msg = result.stderr or result.stdout or ""
            if "cannot find the file" in error_msg.lower() or "system cannot find" in error_msg.lower():
                return True, "Auto-start not found (already removed)"
            else:
                return False, f"Failed to remove Task Scheduler task: {error_msg}"

    except subprocess.TimeoutExpired:
        return False, "Failed to remove auto-start: schtasks command timeout"
    except FileNotFoundError:
        return False, "Failed to remove auto-start: schtasks.exe not found (Windows only)"
    except Exception as e:
        return False, f"Failed to remove auto-start for Windows: {str(e)}"


# ============================================================================
# Cross-Platform Auto-start Manager
# ============================================================================


def install_autostart(script_path: str, config: Dict[str, str]) -> Tuple[bool, str]:
    """
    Install auto-start for current platform.

    Args:
        script_path: Path to worker script
        config: Worker configuration with ws_url and token

    Returns:
        Tuple of (success, message)
    """
    current_platform = detect_platform()

    if current_platform == "mac":
        return install_launchd_plist(script_path, config)
    elif current_platform == "linux":
        return install_systemd_service(script_path, config)
    elif current_platform == "windows":
        return install_task_scheduler_task(script_path, config)
    else:
        return False, f"Unsupported platform: {platform.system()}"


def remove_autostart() -> Tuple[bool, str]:
    """
    Remove auto-start for current platform.

    Returns:
        Tuple of (success, message)
    """
    current_platform = detect_platform()

    if current_platform == "mac":
        return remove_launchd_plist()
    elif current_platform == "linux":
        return remove_systemd_service()
    elif current_platform == "windows":
        return remove_task_scheduler_task()
    else:
        return False, f"Unsupported platform: {platform.system()}"


def get_worker_config_for_autostart() -> Optional[Dict[str, str]]:
    """
    Load worker configuration for auto-start setup.

    Returns:
        Configuration dictionary or None if not found
    """
    try:
        # Try to load config directly from ~/.statsbot/config.json
        config_dir = Path.home() / ".statsbot"
        config_file = config_dir / "config.json"

        if not config_file.exists():
            return None

        import json
        return json.loads(config_file.read_text())

    except Exception:
        return None
