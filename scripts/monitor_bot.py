#!/usr/bin/env python3
"""Monitor bot for wrong version or duplicate instances."""
import subprocess
import time

def check_bot_health():
    # Check for multiple processes
    result = subprocess.run(['pgrep', '-f', 'telegram_bot.py'], capture_output=True)
    processes = result.stdout.decode().strip().split('\n') if result.stdout else []

    if len(processes) > 1:
        print(f"⚠️ WARNING: Multiple bot processes detected: {processes}")
    elif len(processes) == 1:
        print(f"✅ Single bot process running: PID {processes[0]}")
    else:
        print("❌ No bot processes found")

    # Check working directory if process exists
    if len(processes) == 1:
        pid = processes[0]
        try:
            cwd_result = subprocess.run(['lsof', '-p', pid], capture_output=True, text=True)
            if 'statistical-modeling-agent' in cwd_result.stdout:
                print("✅ Bot running from correct directory")
            else:
                print("⚠️ Bot not running from statistical-modeling-agent directory")
        except Exception as e:
            print(f"Could not check working directory: {e}")

if __name__ == "__main__":
    while True:
        check_bot_health()
        time.sleep(60)