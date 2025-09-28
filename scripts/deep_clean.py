#!/usr/bin/env python3
"""
Deep Cache Cleanup System for Telegram Bot

This script performs comprehensive cleanup to ensure the bot loads fresh code:
1. Kills all bot processes
2. Clears ALL Python cache
3. Clears pip cache
4. Removes temporary files
5. Verifies clean state
6. Provides restart instructions
"""

import os
import sys
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Tuple, Optional

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

project_root = Path(__file__).parent.parent

class DeepCleaner:
    """Comprehensive cache and process cleaner."""

    def __init__(self):
        self.processes_killed = []
        self.cache_dirs_removed = []
        self.files_removed = []
        self.errors = []

    def find_bot_processes(self) -> List[dict]:
        """Find all bot-related processes."""
        print("🔍 Searching for bot processes...")
        processes = []

        if not PSUTIL_AVAILABLE:
            print("  ⚠️ psutil not available - manual process check required")
            print("  💡 Use: ps aux | grep python | grep -E '(telegram|bot)'")
            return []

        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd', 'create_time']):
                try:
                    if proc.info['name'] in ['python', 'python3']:
                        cmdline = ' '.join(proc.info['cmdline'] or [])

                        # Look for bot-related keywords
                        bot_keywords = [
                            'telegram_bot.py',
                            'run_bot.py',
                            'test_bot.py',
                            'test_bot_with_parser.py',
                            'start_bot.sh',
                            'statistical-modeling-agent'
                        ]

                        if any(keyword in cmdline for keyword in bot_keywords):
                            processes.append({
                                'pid': proc.info['pid'],
                                'cmdline': cmdline,
                                'cwd': proc.info.get('cwd', 'unknown'),
                                'create_time': proc.info.get('create_time', 0)
                            })

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            self.errors.append(f"Process scan error: {e}")

        print(f"  🤖 Found {len(processes)} bot processes")
        for proc in processes:
            age = time.time() - proc['create_time']
            print(f"    - PID {proc['pid']}: {age:.0f}s old")
            print(f"      Dir: {proc['cwd']}")
            print(f"      Cmd: {proc['cmdline'][:80]}...")

        return processes

    def kill_bot_processes(self, processes: List[dict]) -> bool:
        """Kill all bot processes gracefully."""
        if not processes:
            print("  ✅ No bot processes to kill")
            return True

        print(f"🛑 Terminating {len(processes)} bot processes...")

        for proc_info in processes:
            pid = proc_info['pid']
            try:
                if PSUTIL_AVAILABLE:
                    proc = psutil.Process(pid)
                    print(f"  📤 Terminating PID {pid}...")
                    proc.terminate()

                    # Wait for graceful termination
                    try:
                        proc.wait(timeout=5)
                        print(f"    ✅ PID {pid} terminated gracefully")
                        self.processes_killed.append(pid)
                    except psutil.TimeoutExpired:
                        print(f"    ⚠️ PID {pid} didn't terminate, force killing...")
                        proc.kill()
                        print(f"    ✅ PID {pid} force killed")
                        self.processes_killed.append(pid)
                else:
                    # Fallback without psutil
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(2)
                    try:
                        os.kill(pid, 0)  # Check if still alive
                        os.kill(pid, signal.SIGKILL)  # Force kill
                    except ProcessLookupError:
                        pass  # Process already dead
                    self.processes_killed.append(pid)

            except Exception as e:
                self.errors.append(f"Failed to kill PID {pid}: {e}")
                print(f"    ❌ Failed to kill PID {pid}: {e}")

        return len(self.errors) == 0

    def clear_python_cache(self) -> bool:
        """Clear all Python cache files and directories."""
        print("🗑️ Clearing Python cache...")

        # Find all __pycache__ directories
        pycache_dirs = list(project_root.rglob("__pycache__"))

        # Find all .pyc files
        pyc_files = list(project_root.rglob("*.pyc"))

        # Find all .pyo files (optimized bytecode)
        pyo_files = list(project_root.rglob("*.pyo"))

        total_items = len(pycache_dirs) + len(pyc_files) + len(pyo_files)
        print(f"  📂 Found {len(pycache_dirs)} cache dirs, {len(pyc_files)} .pyc files, {len(pyo_files)} .pyo files")

        # Remove cache directories
        for cache_dir in pycache_dirs:
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
                self.cache_dirs_removed.append(str(cache_dir))
                print(f"    🗂️ Removed {cache_dir.relative_to(project_root)}")
            except Exception as e:
                self.errors.append(f"Failed to remove {cache_dir}: {e}")

        # Remove .pyc files
        for pyc_file in pyc_files + pyo_files:
            try:
                pyc_file.unlink(missing_ok=True)
                self.files_removed.append(str(pyc_file))
                print(f"    📄 Removed {pyc_file.relative_to(project_root)}")
            except Exception as e:
                self.errors.append(f"Failed to remove {pyc_file}: {e}")

        print(f"  ✅ Cleared {len(self.cache_dirs_removed)} directories, {len(self.files_removed)} files")
        return len(self.errors) == 0

    def clear_pip_cache(self) -> bool:
        """Clear pip cache to ensure fresh package imports."""
        print("📦 Clearing pip cache...")

        try:
            # Get pip cache directory
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'cache', 'dir'
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0:
                cache_dir = Path(result.stdout.strip())
                if cache_dir.exists():
                    print(f"  🗂️ Pip cache: {cache_dir}")

                    # Clear the cache
                    clear_result = subprocess.run([
                        sys.executable, '-m', 'pip', 'cache', 'purge'
                    ], capture_output=True, text=True, timeout=30)

                    if clear_result.returncode == 0:
                        print("  ✅ Pip cache cleared")
                        return True
                    else:
                        self.errors.append(f"Pip cache clear failed: {clear_result.stderr}")
                else:
                    print("  ℹ️ No pip cache found")
                    return True
            else:
                self.errors.append(f"Pip cache dir failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            self.errors.append("Pip cache operations timed out")
        except Exception as e:
            self.errors.append(f"Pip cache error: {e}")

        return len(self.errors) == 0

    def clear_temp_files(self) -> bool:
        """Clear temporary files that might interfere."""
        print("🧹 Clearing temporary files...")

        temp_patterns = [
            "*.tmp",
            "*.temp",
            "*.log",
            ".DS_Store",
            "Thumbs.db",
            "*.swp",
            "*.swo",
            "*~"
        ]

        removed_count = 0
        for pattern in temp_patterns:
            for temp_file in project_root.rglob(pattern):
                try:
                    # Skip important logs and config files
                    if any(keep in str(temp_file) for keep in ['.env', 'requirements', 'CLAUDE.md']):
                        continue

                    temp_file.unlink(missing_ok=True)
                    removed_count += 1
                    print(f"    🗑️ Removed {temp_file.relative_to(project_root)}")
                except Exception as e:
                    self.errors.append(f"Failed to remove {temp_file}: {e}")

        print(f"  ✅ Removed {removed_count} temporary files")
        return True

    def verify_clean_state(self) -> Tuple[bool, List[str]]:
        """Verify that cleanup was successful."""
        print("🔍 Verifying clean state...")
        issues = []

        # Check for remaining cache
        remaining_cache = list(project_root.rglob("__pycache__")) + list(project_root.rglob("*.pyc"))
        if remaining_cache:
            issues.append(f"Found {len(remaining_cache)} remaining cache files")

        # Check for remaining processes
        remaining_processes = self.find_bot_processes()
        if remaining_processes:
            issues.append(f"Found {len(remaining_processes)} remaining bot processes")

        # Check handlers.py content
        handlers_path = project_root / "src" / "bot" / "handlers.py"
        if handlers_path.exists():
            content = handlers_path.read_text()
            if "I'm currently under development" in content:
                issues.append("handlers.py still contains Development Mode text")
            if "DataLoader v2.0" not in content:
                issues.append("handlers.py missing DataLoader integration")
        else:
            issues.append("handlers.py file not found")

        if not issues:
            print("  ✅ Clean state verified")
        else:
            print("  ⚠️ Issues found:")
            for issue in issues:
                print(f"    - {issue}")

        return len(issues) == 0, issues

    def create_version_file(self) -> bool:
        """Create a version file to track when cleanup occurred."""
        print("📝 Creating version tracking...")

        try:
            version_file = project_root / "VERSION"
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            version_content = f"""# Statistical Modeling Agent Version
CLEANUP_TIMESTAMP={timestamp}
DEEP_CLEAN_VERSION=1.0
DATALOADER_INTEGRATION=v2.0
CACHE_CLEARED=true
PROCESSES_KILLED={len(self.processes_killed)}
"""
            version_file.write_text(version_content)
            print(f"  ✅ Version file created: {version_file}")
            return True

        except Exception as e:
            self.errors.append(f"Version file creation failed: {e}")
            return False

    def run_full_cleanup(self) -> bool:
        """Run complete cleanup process."""
        print("🧹 DEEP CLEANUP STARTING")
        print("=" * 50)

        start_time = time.time()

        # Step 1: Find and kill processes
        processes = self.find_bot_processes()
        if not self.kill_bot_processes(processes):
            print("⚠️ Process termination had issues, continuing...")

        # Step 2: Clear Python cache
        if not self.clear_python_cache():
            print("⚠️ Python cache clearing had issues, continuing...")

        # Step 3: Clear pip cache
        if not self.clear_pip_cache():
            print("⚠️ Pip cache clearing had issues, continuing...")

        # Step 4: Clear temp files
        self.clear_temp_files()

        # Step 5: Create version file
        self.create_version_file()

        # Step 6: Verify clean state
        is_clean, issues = self.verify_clean_state()

        # Summary
        duration = time.time() - start_time
        print("\n" + "=" * 50)
        print("📊 CLEANUP SUMMARY")
        print("=" * 50)
        print(f"⏱️ Duration: {duration:.2f} seconds")
        print(f"🛑 Processes killed: {len(self.processes_killed)}")
        print(f"🗂️ Cache dirs removed: {len(self.cache_dirs_removed)}")
        print(f"📄 Files removed: {len(self.files_removed)}")
        print(f"❌ Errors: {len(self.errors)}")

        if self.errors:
            print("\n🚨 ERRORS ENCOUNTERED:")
            for error in self.errors:
                print(f"  - {error}")

        print(f"\n{'✅ CLEANUP SUCCESSFUL' if is_clean else '⚠️ CLEANUP INCOMPLETE'}")

        if not is_clean:
            print("\n🔧 REMAINING ISSUES:")
            for issue in issues:
                print(f"  - {issue}")

        # Restart instructions
        print("\n🚀 RESTART INSTRUCTIONS")
        print("=" * 30)
        print("1. cd to project directory")
        print("2. source venv/bin/activate")
        print("3. python src/bot/telegram_bot.py")
        print("\nOr use: ./start_bot.sh")

        print("\n🔧 WATCH FOR THESE LOGS:")
        print("  - '🔧 HANDLERS DIAGNOSTIC: DataLoader v2.0 integration ACTIVE'")
        print("  - '🔧 DATALOADER IMPORT: Success'")
        print("  - '🔧 MESSAGE HANDLER DIAGNOSTIC: Processing user message'")

        print("\n✅ Upload a CSV file to test DataLoader integration!")

        return is_clean and len(self.errors) == 0


def main():
    """Main cleanup execution."""
    print("🤖 STATISTICAL MODELING AGENT - DEEP CLEANUP")
    print("=" * 60)

    try:
        cleaner = DeepCleaner()
        success = cleaner.run_full_cleanup()
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️ Cleanup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Cleanup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()