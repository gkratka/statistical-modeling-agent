# Auto-start Support Implementation Summary

**Task**: Task 6.0 - Auto-start Support (Mac/Linux/Windows)
**Date**: 2025-11-30
**Status**: ✅ Complete (6/7 sub-tasks)

## Overview

Implemented comprehensive auto-start support for the StatsBot Local Worker, enabling users to configure the worker to start automatically when their system boots. The implementation uses platform-specific mechanisms for maximum reliability and native OS integration.

## Implementation Details

### Files Created

1. **`worker/autostart.py`** (562 lines)
   - Platform detection (`detect_platform()`)
   - Mac launchd plist generation and installation
   - Linux systemd user service generation and installation
   - Windows Task Scheduler task creation via schtasks.exe
   - Cross-platform auto-start manager
   - Configuration persistence helper

2. **`tests/unit/test_autostart.py`** (613 lines)
   - 40 comprehensive tests covering all platforms
   - Mac launchd tests (8 tests)
   - Linux systemd tests (8 tests)
   - Windows Task Scheduler tests (9 tests)
   - Platform detection tests (4 tests)
   - Cross-platform manager tests (7 tests)
   - Integration tests (4 tests)
   - All tests passing ✅

### Files Modified

1. **`worker/statsbot_worker.py`**
   - Updated `main()` function to handle `--autostart` flag
   - Added support for `--autostart on` (install) and `--autostart off` (remove)
   - Made `--token` optional when using `--autostart off`
   - Integrated autostart module import and execution
   - Enhanced error handling for auto-start operations

2. **`tasks/tasks-local-worker.md`**
   - Marked tasks 6.1-6.5 and 6.7 as complete
   - Task 6.6 (`/worker autostart` Telegram command) remains pending

## Platform-Specific Implementations

### Mac (launchd)

**Location**: `~/Library/LaunchAgents/com.statsbot.worker.plist`

**Features**:
- RunAtLoad: Starts worker on user login
- KeepAlive: Automatically restarts on failure
- Logging: stdout/stderr to `~/.statsbot/logs/`
- Working directory: `~/.statsbot/`
- Environment variables set correctly

**Example plist**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" ...>
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.statsbot.worker</string>
    <key>ProgramArguments</key>
    <array>
        <string>/usr/bin/python3</string>
        <string>/path/to/statsbot_worker.py</string>
        <string>--token=TOKEN</string>
        <string>--ws-url=WS_URL</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
</dict>
</plist>
```

### Linux (systemd)

**Location**: `~/.config/systemd/user/statsbot-worker.service`

**Features**:
- User service (no sudo required)
- Restart on failure with 10-second delay
- Network dependency (waits for network)
- Logging: stdout/stderr to `~/.statsbot/logs/`
- Working directory: `~/.statsbot/`

**Example service file**:
```ini
[Unit]
Description=StatsBot Local Worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /path/to/statsbot_worker.py --token=TOKEN --ws-url=WS_URL
Restart=on-failure
RestartSec=10

[Install]
WantedBy=default.target
```

**Manual commands** (optional):
```bash
systemctl --user enable statsbot-worker.service
systemctl --user start statsbot-worker.service
systemctl --user status statsbot-worker.service
```

### Windows (Task Scheduler)

**Task Name**: `StatsBot-Worker`

**Features**:
- Logon trigger: Starts on user login
- Network availability check
- Restart on failure (3 attempts with 1-minute interval)
- No battery restrictions
- Working directory: `%USERPROFILE%\.statsbot`

**Example XML** (abbreviated):
```xml
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="...">
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>python</Command>
      <Arguments>"/path/to/statsbot_worker.py" --token=TOKEN --ws-url=WS_URL</Arguments>
    </Exec>
  </Actions>
  <Settings>
    <StartWhenAvailable>true</StartWenAvailable>
    <RestartOnFailure>
      <Interval>PT1M</Interval>
      <Count>3</Count>
    </RestartOnFailure>
  </Settings>
</Task>
```

**Manual commands** (optional):
```cmd
schtasks /Run /TN "StatsBot-Worker"
schtasks /Query /TN "StatsBot-Worker"
```

## Usage Examples

### Install Auto-start

**First-time setup** (requires token):
```bash
python3 statsbot_worker.py --token=YOUR_TOKEN --autostart
```

**Subsequent installs** (uses saved config):
```bash
python3 statsbot_worker.py --autostart on
```

**Success output**:
```
StatsBot Worker v1.0.0
Machine: my-macbook

Installing auto-start configuration...

✅ Auto-start installed successfully for Mac (launchd)
Plist location: /Users/username/Library/LaunchAgents/com.statsbot.worker.plist

📝 Note: Auto-start configuration has been created.
   The worker will start automatically when you log in.
```

### Remove Auto-start

```bash
python3 statsbot_worker.py --autostart off
```

**Success output**:
```
StatsBot Worker v1.0.0
Machine: my-macbook

Removing auto-start configuration...

✅ Auto-start removed successfully for Mac (launchd)
```

## Configuration Persistence

Worker configuration is saved to `~/.statsbot/config.json`:

```json
{
  "ws_url": "wss://statsbot.example.com/ws",
  "token": "uuid-token-here",
  "machine_id": "hostname"
}
```

This allows:
1. Auto-start without needing to specify token again
2. Token reuse across worker restarts
3. Machine identifier persistence

## Security Considerations

1. **Token Storage**: Tokens are stored in `~/.statsbot/config.json`
   - File permissions: User-readable only (default)
   - Location: User home directory (not system-wide)
   - Recommendation: Use file permissions to restrict access

2. **Auto-start Scope**: User-level only
   - Mac: User LaunchAgents (not system-wide LaunchDaemons)
   - Linux: User systemd service (not system service)
   - Windows: User task (not system task)

3. **Script Permissions**: Auto-start configs reference absolute script path
   - Validates script exists at installation time
   - Uses resolved absolute path (no symlink attacks)

## Test Coverage

**40 tests, 100% passing**

- Mac launchd: 8 tests
  - Plist structure validation
  - Label, program arguments, RunAtLoad, KeepAlive
  - Installation and removal workflows
  - Error handling

- Linux systemd: 8 tests
  - Service file structure validation
  - Description, ExecStart, restart policy, WantedBy
  - Installation and removal workflows
  - Error handling

- Windows Task Scheduler: 9 tests
  - XML structure validation
  - Registration info, triggers, actions, settings
  - Installation and removal (success/failure)
  - schtasks error handling

- Platform detection: 4 tests
  - Mac (Darwin), Linux, Windows detection
  - Unsupported platform handling

- Cross-platform manager: 7 tests
  - Platform-specific routing
  - Install/remove delegation
  - Unsupported platform errors

- Integration: 4 tests
  - Complete install/remove workflows
  - Config persistence and loading
  - Error handling for missing config

## Completed Sub-tasks

- ✅ 6.1: `--autostart` flag parsing in worker script
- ✅ 6.2: Mac launchd plist creation
- ✅ 6.3: Linux systemd user service creation
- ✅ 6.4: Windows Task Scheduler task creation
- ✅ 6.5: `--autostart off` to remove auto-start
- ⏸️ 6.6: `/worker autostart` Telegram command (pending)
- ✅ 6.7: Comprehensive test suite (40 tests)

## Pending Work

**Task 6.6**: Add `/worker autostart` command in Telegram

This would provide users with:
- Platform-specific instructions
- Installation command template with their token
- Removal instructions
- Troubleshooting guidance

**Example interaction**:
```
User: /worker autostart

Bot: 📋 Auto-start Setup Instructions

Platform: Mac (macOS)

To install auto-start:
  python3 statsbot_worker.py --token=abc123 --autostart

To remove auto-start:
  python3 statsbot_worker.py --autostart off

The worker will start automatically when you log in.
```

## Architecture Benefits

1. **Self-contained**: All auto-start logic in `worker/autostart.py`
2. **Platform-agnostic**: Single API (`install_autostart()`, `remove_autostart()`)
3. **Testable**: Comprehensive test coverage with mocking
4. **Production-ready**: Error handling, logging, user-friendly messages
5. **Maintainable**: Clean separation of concerns, well-documented

## Related Files

- **Implementation**: `/Users/gkratka/Documents/statistical-modeling-agent/worker/autostart.py`
- **Worker Script**: `/Users/gkratka/Documents/statistical-modeling-agent/worker/statsbot_worker.py`
- **Tests**: `/Users/gkratka/Documents/statistical-modeling-agent/tests/unit/test_autostart.py`
- **Task Tracker**: `/Users/gkratka/Documents/statistical-modeling-agent/tasks/tasks-local-worker.md`

## Conclusion

Task 6.0 is 86% complete (6/7 sub-tasks). The core auto-start functionality is fully implemented and tested across all platforms. Only the Telegram command helper (task 6.6) remains pending. The implementation follows TDD principles with tests written first, ensuring production-quality code with comprehensive coverage.
