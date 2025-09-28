#!/usr/bin/env python3
"""
Bot Verification System

This script verifies the bot is ready to run with correct DataLoader integration:
1. Checks handlers.py content
2. Verifies DataLoader imports
3. Tests file checksums
4. Validates environment
5. Provides pre-start diagnostics
"""

import sys
import hashlib
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class BotVerifier:
    """Comprehensive bot verification system."""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.errors = []

    def check_file_integrity(self) -> bool:
        """Verify critical files exist and have expected content."""
        print("🔍 Checking file integrity...")

        critical_files = {
            "src/bot/handlers.py": {
                "must_contain": [
                    "from src.processors.data_loader import DataLoader",
                    "loader.load_from_telegram",
                    "HANDLERS DIAGNOSTIC",
                    "MESSAGE HANDLER DIAGNOSTIC"
                ],
                "must_not_contain": [
                    "I'm currently under development. Soon I'll",
                    "File upload handling is under development",
                    "For now, I'm confirming that I can receive"
                ]
            },
            "src/processors/data_loader.py": {
                "must_contain": [
                    "class DataLoader",
                    "async def load_from_telegram",
                    "def get_data_summary"
                ],
                "must_not_contain": []
            },
            "src/bot/telegram_bot.py": {
                "must_contain": [
                    "from src.bot.handlers import",
                    "document_handler",
                    "message_handler"
                ],
                "must_not_contain": []
            }
        }

        all_passed = True

        for file_path, checks in critical_files.items():
            full_path = project_root / file_path
            print(f"  📄 Checking {file_path}...")

            if not full_path.exists():
                print(f"    ❌ File missing: {file_path}")
                self.errors.append(f"Missing critical file: {file_path}")
                all_passed = False
                continue

            content = full_path.read_text()

            # Check required content
            for required in checks["must_contain"]:
                if required not in content:
                    print(f"    ❌ Missing required text: '{required[:50]}...'")
                    self.errors.append(f"{file_path} missing: {required}")
                    all_passed = False
                else:
                    print(f"    ✅ Found: '{required[:30]}...'")

            # Check forbidden content
            for forbidden in checks["must_not_contain"]:
                if forbidden in content:
                    print(f"    ❌ Found forbidden text: '{forbidden[:50]}...'")
                    self.errors.append(f"{file_path} contains old code: {forbidden}")
                    all_passed = False

        if all_passed:
            print("  ✅ All file integrity checks passed")
            self.checks_passed += 1
        else:
            print("  ❌ File integrity checks failed")
            self.checks_failed += 1

        return all_passed

    def check_imports(self) -> bool:
        """Verify all required imports work correctly."""
        print("📦 Checking imports...")

        import_tests = [
            ("DataLoader", "from src.processors.data_loader import DataLoader"),
            ("Handlers", "from src.bot.handlers import document_handler, message_handler"),
            ("Exceptions", "from src.utils.exceptions import ValidationError, DataError"),
            ("Pandas", "import pandas as pd"),
            ("Telegram", "from telegram.ext import Application")
        ]

        all_passed = True

        for test_name, import_statement in import_tests:
            print(f"  🔗 Testing {test_name}...")
            try:
                exec(import_statement)
                print(f"    ✅ {test_name} import successful")
            except ImportError as e:
                print(f"    ❌ {test_name} import failed: {e}")
                self.errors.append(f"Import error for {test_name}: {e}")
                all_passed = False
            except Exception as e:
                print(f"    ❌ {test_name} import error: {e}")
                self.errors.append(f"Unexpected error for {test_name}: {e}")
                all_passed = False

        if all_passed:
            print("  ✅ All import checks passed")
            self.checks_passed += 1
        else:
            print("  ❌ Import checks failed")
            self.checks_failed += 1

        return all_passed

    def check_dataloader_functionality(self) -> bool:
        """Test DataLoader core functionality."""
        print("🔬 Testing DataLoader functionality...")

        try:
            from src.processors.data_loader import DataLoader
            import pandas as pd
            import tempfile

            loader = DataLoader()
            print(f"  ✅ DataLoader created (max size: {loader.MAX_FILE_SIZE / 1024 / 1024:.1f}MB)")

            # Test DataFrame validation
            test_df = pd.DataFrame({
                'test_col1': [1, 2, 3],
                'test_col2': ['a', 'b', 'c']
            })

            metadata = loader._validate_dataframe(test_df, "test.csv")
            print(f"  ✅ DataFrame validation works (shape: {metadata['shape']})")

            # Test summary generation
            summary = loader.get_data_summary(test_df, metadata)
            if "Data Successfully Loaded" in summary:
                print("  ✅ Summary generation works")
            else:
                print("  ⚠️ Summary generation may have issues")
                self.warnings.append("Summary format may be incorrect")

            self.checks_passed += 1
            return True

        except Exception as e:
            print(f"  ❌ DataLoader functionality test failed: {e}")
            self.errors.append(f"DataLoader functionality error: {e}")
            self.checks_failed += 1
            return False

    def check_environment(self) -> bool:
        """Check environment variables and configuration."""
        print("🌍 Checking environment...")

        env_checks = [
            (".env file", project_root / ".env"),
            ("requirements.txt", project_root / "requirements.txt"),
            ("venv directory", project_root / "venv"),
            ("src directory", project_root / "src")
        ]

        all_passed = True

        for check_name, path in env_checks:
            print(f"  📂 Checking {check_name}...")
            if path.exists():
                print(f"    ✅ {check_name} found")
            else:
                print(f"    ❌ {check_name} missing: {path}")
                self.errors.append(f"Missing {check_name}: {path}")
                all_passed = False

        # Check .env content
        env_file = project_root / ".env"
        if env_file.exists():
            content = env_file.read_text()
            if "TELEGRAM_BOT_TOKEN" in content:
                print("    ✅ Bot token configured")
            else:
                print("    ⚠️ Bot token may not be configured")
                self.warnings.append("Check TELEGRAM_BOT_TOKEN in .env")

        if all_passed:
            self.checks_passed += 1
        else:
            self.checks_failed += 1

        return all_passed

    def check_version_markers(self) -> bool:
        """Check for version markers and diagnostic logging."""
        print("🏷️ Checking version markers...")

        handlers_file = project_root / "src" / "bot" / "handlers.py"
        if not handlers_file.exists():
            print("  ❌ handlers.py not found")
            self.errors.append("handlers.py missing")
            self.checks_failed += 1
            return False

        content = handlers_file.read_text()

        version_markers = [
            "HANDLERS DIAGNOSTIC",
            "DATALOADER IMPORT",
            "MESSAGE HANDLER DIAGNOSTIC",
            "DataLoader v2.0"
        ]

        found_markers = 0
        for marker in version_markers:
            if marker in content:
                print(f"  ✅ Found marker: {marker}")
                found_markers += 1
            else:
                print(f"  ⚠️ Missing marker: {marker}")

        if found_markers >= 3:
            print(f"  ✅ Version markers sufficient ({found_markers}/{len(version_markers)})")
            self.checks_passed += 1
            return True
        else:
            print(f"  ❌ Insufficient version markers ({found_markers}/{len(version_markers)})")
            self.warnings.append("Add more diagnostic markers for debugging")
            self.checks_failed += 1
            return False

    def generate_file_checksums(self) -> Dict[str, str]:
        """Generate checksums for critical files."""
        print("🔐 Generating file checksums...")

        critical_files = [
            "src/bot/handlers.py",
            "src/processors/data_loader.py",
            "src/bot/telegram_bot.py"
        ]

        checksums = {}
        for file_path in critical_files:
            full_path = project_root / file_path
            if full_path.exists():
                content = full_path.read_bytes()
                checksum = hashlib.sha256(content).hexdigest()[:16]
                checksums[file_path] = checksum
                print(f"  📄 {file_path}: {checksum}")
            else:
                checksums[file_path] = "MISSING"
                print(f"  ❌ {file_path}: MISSING")

        return checksums

    def create_diagnostic_report(self, checksums: Dict[str, str]) -> str:
        """Create a diagnostic report for troubleshooting."""
        report_lines = [
            "# Bot Verification Report",
            f"Generated: {Path(__file__).name}",
            "",
            "## Summary",
            f"✅ Checks passed: {self.checks_passed}",
            f"❌ Checks failed: {self.checks_failed}",
            f"⚠️ Warnings: {len(self.warnings)}",
            f"🚨 Errors: {len(self.errors)}",
            "",
            "## File Checksums",
        ]

        for file_path, checksum in checksums.items():
            report_lines.append(f"- {file_path}: {checksum}")

        if self.warnings:
            report_lines.extend(["", "## Warnings"])
            for warning in self.warnings:
                report_lines.append(f"- {warning}")

        if self.errors:
            report_lines.extend(["", "## Errors"])
            for error in self.errors:
                report_lines.append(f"- {error}")

        report_lines.extend([
            "",
            "## Expected Bot Logs",
            "When uploading a file, look for:",
            "- '🔧 HANDLERS DIAGNOSTIC: DataLoader v2.0 integration ACTIVE'",
            "- '🔧 DATALOADER IMPORT: Success'",
            "- '📁 **Processing your file...**'",
            "",
            "When sending a message, look for:",
            "- '🔧 MESSAGE HANDLER DIAGNOSTIC: Processing user message'",
            "- No 'Development Mode' text"
        ])

        return "\n".join(report_lines)

    def run_full_verification(self) -> bool:
        """Run complete verification process."""
        print("🔍 BOT VERIFICATION STARTING")
        print("=" * 50)

        # Run all checks
        checks = [
            ("File Integrity", self.check_file_integrity),
            ("Imports", self.check_imports),
            ("DataLoader Functionality", self.check_dataloader_functionality),
            ("Environment", self.check_environment),
            ("Version Markers", self.check_version_markers)
        ]

        for check_name, check_func in checks:
            print(f"\n🔬 {check_name}")
            print("-" * 30)
            check_func()

        # Generate checksums
        print(f"\n🔐 File Checksums")
        print("-" * 20)
        checksums = self.generate_file_checksums()

        # Create diagnostic report
        report = self.create_diagnostic_report(checksums)
        report_file = project_root / "diagnostic_report.md"
        report_file.write_text(report)

        # Summary
        print("\n" + "=" * 50)
        print("📊 VERIFICATION SUMMARY")
        print("=" * 50)
        print(f"✅ Passed: {self.checks_passed}")
        print(f"❌ Failed: {self.checks_failed}")
        print(f"⚠️ Warnings: {len(self.warnings)}")

        success = self.checks_failed == 0

        if success:
            print("\n🎉 BOT READY TO START!")
            print("🚀 Run: python src/bot/telegram_bot.py")
        else:
            print("\n⚠️ BOT NOT READY - FIX ERRORS FIRST")
            print("🔧 See diagnostic_report.md for details")

        print(f"\n📄 Detailed report: {report_file}")
        return success


def main():
    """Main verification execution."""
    try:
        verifier = BotVerifier()
        success = verifier.run_full_verification()
        sys.exit(0 if success else 1)

    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()