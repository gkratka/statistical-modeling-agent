"""Integration tests for bot process lifecycle management."""
import pytest
import subprocess
import time
import os
from pathlib import Path


class TestBotInstanceManagement:
    """TDD tests for bot process lifecycle management."""

    def test_pid_file_created_on_startup(self):
        """Test 1: Bot creates PID file at startup."""
        # ARRANGE
        pid_file = Path(".bot.pid")
        if pid_file.exists():
            pid_file.unlink()

        # ACT
        proc = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)

        # ASSERT
        assert pid_file.exists(), "PID file not created"
        pid = int(pid_file.read_text())
        assert pid == proc.pid, "PID file contains wrong PID"

        # CLEANUP
        proc.terminate()
        proc.wait()

    def test_second_instance_refuses_to_start(self):
        """Test 2: Second bot instance detects existing PID and exits."""
        # ARRANGE
        proc1 = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)

        # ACT
        proc2 = subprocess.Popen(
            ["python3", "src/bot/telegram_bot.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        stdout, stderr = proc2.communicate(timeout=5)

        # ASSERT
        assert proc2.returncode != 0, "Second instance should fail"
        assert b"already running" in stderr.lower(), "Wrong error message"

        # CLEANUP
        proc1.terminate()
        proc1.wait()

    def test_pid_file_cleanup_on_graceful_shutdown(self):
        """Test 3: PID file removed on clean shutdown."""
        # ARRANGE
        pid_file = Path(".bot.pid")
        proc = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)
        assert pid_file.exists()

        # ACT
        proc.terminate()
        proc.wait(timeout=10)

        # ASSERT
        assert not pid_file.exists(), "PID file not cleaned up"

    def test_stale_pid_detection_and_override(self):
        """Test 4: Stale PID (dead process) allows new instance."""
        # ARRANGE
        pid_file = Path(".bot.pid")
        pid_file.write_text("99999")  # Non-existent PID

        # ACT
        proc = subprocess.Popen(["python3", "src/bot/telegram_bot.py"])
        time.sleep(3)

        # ASSERT
        assert proc.poll() is None, "Bot should start with stale PID"
        new_pid = int(pid_file.read_text())
        assert new_pid == proc.pid, "PID file not updated"

        # CLEANUP
        proc.terminate()
        proc.wait()

    def test_clean_startup_script_kills_zombies(self):
        """Test 5: start_bot_clean.sh cleans up zombie processes."""
        # ARRANGE - Create 3 zombie background processes
        for i in range(3):
            subprocess.Popen(
                ["python3", "src/bot/telegram_bot.py"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        time.sleep(2)

        # ACT
        result = subprocess.run(
            ["./scripts/start_bot_clean.sh"],
            capture_output=True,
            text=True
        )

        # ASSERT
        assert result.returncode == 0, "Startup script failed"
        assert "TEST 2 PASSED" in result.stdout, "Cleanup verification failed"

        # Verify only 1 process running
        ps_result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True
        )
        bot_processes = [
            line for line in ps_result.stdout.split("\n")
            if "telegram_bot" in line and "grep" not in line
        ]
        assert len(bot_processes) == 1, f"Expected 1 process, found {len(bot_processes)}"
