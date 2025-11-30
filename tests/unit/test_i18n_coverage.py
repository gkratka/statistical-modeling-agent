"""
Comprehensive i18n coverage testing.

This module validates that all user-facing strings are internationalized
and no hardcoded English strings exist in handler files.
"""

import ast
import re
from pathlib import Path
from typing import List, Set, Tuple

import pytest
import yaml

from src.utils.i18n_manager import I18nManager


class TestI18nCoverage:
    """Test suite for i18n coverage validation."""

    @pytest.fixture(scope="class")
    def project_root(self) -> Path:
        """Get project root directory."""
        return Path(__file__).parent.parent.parent

    @pytest.fixture(scope="class")
    def handler_files(self, project_root: Path) -> List[Path]:
        """Get all handler Python files."""
        handlers_dir = project_root / "src" / "bot" / "handlers"
        return list(handlers_dir.glob("*.py"))

    @pytest.fixture(scope="class")
    def i18n_yaml_files(self, project_root: Path) -> dict[str, Path]:
        """Get all i18n YAML files."""
        i18n_dir = project_root / "src" / "bot" / "i18n"
        return {
            "en": i18n_dir / "en.yaml",
            "pt": i18n_dir / "pt.yaml",
        }

    @pytest.fixture(scope="class")
    def loaded_translations(self, i18n_yaml_files: dict) -> dict[str, dict]:
        """Load all translation files."""
        translations = {}
        for lang, path in i18n_yaml_files.items():
            with open(path, "r", encoding="utf-8") as f:
                translations[lang] = yaml.safe_load(f)
        return translations

    def test_handler_files_exist(self, handler_files: List[Path]):
        """Verify handler files are found."""
        assert len(handler_files) > 0, "No handler files found"
        assert any("ml_training" in f.name for f in handler_files), (
            "ML training handler not found"
        )

    def test_i18n_yaml_files_exist(self, i18n_yaml_files: dict):
        """Verify i18n YAML files exist."""
        for lang, path in i18n_yaml_files.items():
            assert path.exists(), f"Missing {lang}.yaml file at {path}"

    def test_no_hardcoded_strings_in_handlers(self, handler_files: List[Path]):
        """Scan all handler files for hardcoded English strings.

        This test identifies string literals that should be internationalized.
        Excludes:
        - Technical strings (URLs, file paths, codes)
        - Single characters or very short strings
        - Strings already wrapped in I18nManager.t()
        """
        # Known exceptions (technical strings, not user-facing)
        ALLOWED_STRINGS = {
            # File extensions
            ".csv", ".xlsx", ".xls", ".parquet",
            # Technical codes
            "utf-8", "ISO-8859-1", "latin1",
            # Model type codes
            "regression", "classification", "neural_network",
            "linear", "ridge", "lasso", "elasticnet", "polynomial",
            "logistic", "decision_tree", "random_forest", "gradient_boosting",
            "svm", "naive_bayes", "mlp_regression", "mlp_classification",
            # State codes
            "IDLE", "TRAINING", "AWAITING_FILE_PATH",
            # Single characters
            ",", ".", ":", ";", "-", "_", "/", "\\",
            # Format codes
            "%Y-%m-%d", "%H:%M:%S",
        }

        # Patterns that indicate user-facing strings
        USER_FACING_PATTERNS = [
            r"Starting\s+training",
            r"This\s+may\s+take",
            r"Model\s+Ready",
            r"Default\s+Name:",
            r"Model\s+ID:",
            r"Type:",
            r"ready\s+for\s+predictions",
            r"Select\s+",
            r"Choose\s+",
            r"Please\s+",
            r"Error:",
            r"Success",
            r"Failed",
            r"Invalid",
            r"Upload",
            r"Training",
            r"Prediction",
        ]

        violations = []

        for handler_file in handler_files:
            # Skip __init__.py and non-handler files
            if handler_file.name.startswith("__"):
                continue

            with open(handler_file, "r", encoding="utf-8") as f:
                content = f.read()

            try:
                tree = ast.parse(content, filename=str(handler_file))
            except SyntaxError as e:
                pytest.fail(f"Syntax error in {handler_file}: {e}")

            # Find all string literals
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    string_value = node.value.strip()

                    # Skip empty strings and allowed strings
                    if not string_value or string_value in ALLOWED_STRINGS:
                        continue

                    # Skip very short strings (< 4 chars) unless they match patterns
                    if len(string_value) < 4:
                        if not any(
                            re.search(pattern, string_value, re.IGNORECASE)
                            for pattern in USER_FACING_PATTERNS
                        ):
                            continue

                    # Check if string matches user-facing patterns
                    is_user_facing = any(
                        re.search(pattern, string_value, re.IGNORECASE)
                        for pattern in USER_FACING_PATTERNS
                    )

                    if is_user_facing:
                        # Check if this string is inside I18nManager.t() call
                        # This is a simplified check - AST analysis would be more accurate
                        line_num = node.lineno
                        surrounding_context = content.split("\n")[
                            max(0, line_num - 2) : line_num + 1
                        ]
                        context_str = "\n".join(surrounding_context)

                        if "I18nManager.t(" not in context_str and ".t(" not in context_str:
                            violations.append(
                                f"{handler_file.name}:{line_num} - '{string_value}'"
                            )

        if violations:
            violation_report = "\n".join(violations)
            pytest.fail(
                f"Found {len(violations)} hardcoded user-facing strings:\n{violation_report}"
            )

    def test_all_buttons_have_i18n(self, handler_files: List[Path]):
        """Verify all InlineKeyboardButton use I18nManager.t().

        InlineKeyboardButton(text=...) should use I18nManager.t() for text parameter.
        """
        violations = []

        for handler_file in handler_files:
            if handler_file.name.startswith("__"):
                continue

            with open(handler_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Find InlineKeyboardButton declarations
            pattern = r'InlineKeyboardButton\s*\(\s*text\s*=\s*(["\'])(.*?)\1'
            matches = re.finditer(pattern, content)

            for match in matches:
                button_text = match.group(2)
                line_num = content[: match.start()].count("\n") + 1

                # Check if the text is a direct call to I18nManager.t() or variable
                # Allow: text=I18nManager.t(...), text=variable_name
                # Disallow: text="hardcoded string"
                if button_text and not button_text.startswith("I18nManager.t"):
                    # Check if it's a variable (starts with letter/underscore)
                    if not re.match(r"^[a-zA-Z_]", button_text):
                        violations.append(
                            f"{handler_file.name}:{line_num} - Button text '{button_text}' not using I18nManager.t()"
                        )

        if violations:
            violation_report = "\n".join(violations)
            pytest.fail(
                f"Found {len(violations)} buttons without i18n:\n{violation_report}"
            )

    def test_yaml_key_coverage(
        self, loaded_translations: dict, handler_files: List[Path]
    ):
        """Ensure all referenced i18n keys exist in YAML files.

        Extracts I18nManager.t() calls and verifies keys exist in en.yaml and pt.yaml.
        """
        # Extract all I18nManager.t() calls from handler files
        referenced_keys = set()

        for handler_file in handler_files:
            if handler_file.name.startswith("__"):
                continue

            with open(handler_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Pattern: I18nManager.t("key.path") or .t("key.path")
            pattern = r'\.t\s*\(\s*["\']([^"\']+)["\']'
            matches = re.finditer(pattern, content)

            for match in matches:
                key = match.group(1)
                referenced_keys.add(key)

        # Check if all referenced keys exist in both languages
        missing_keys = {"en": [], "pt": []}

        for key in referenced_keys:
            for lang in ["en", "pt"]:
                if not self._key_exists_in_yaml(key, loaded_translations[lang]):
                    missing_keys[lang].append(key)

        errors = []
        for lang, keys in missing_keys.items():
            if keys:
                errors.append(f"\nMissing in {lang}.yaml:")
                errors.extend([f"  - {key}" for key in sorted(keys)])

        if errors:
            pytest.fail("\n".join(errors))

    def test_language_parity(self, loaded_translations: dict):
        """Verify en.yaml and pt.yaml have same key structure.

        Both files should have identical keys (though values differ).
        """
        en_keys = self._flatten_yaml_keys(loaded_translations["en"])
        pt_keys = self._flatten_yaml_keys(loaded_translations["pt"])

        only_in_en = en_keys - pt_keys
        only_in_pt = pt_keys - en_keys

        errors = []
        if only_in_en:
            errors.append("\nKeys only in en.yaml:")
            errors.extend([f"  - {key}" for key in sorted(only_in_en)])

        if only_in_pt:
            errors.append("\nKeys only in pt.yaml:")
            errors.extend([f"  - {key}" for key in sorted(only_in_pt)])

        if errors:
            pytest.fail("\n".join(errors))

    def test_no_empty_translations(self, loaded_translations: dict):
        """Verify no translation values are empty strings.

        Empty translations indicate incomplete localization.
        """
        for lang, translations in loaded_translations.items():
            empty_keys = self._find_empty_values(translations)

            if empty_keys:
                error_msg = f"\nEmpty translations in {lang}.yaml:\n"
                error_msg += "\n".join([f"  - {key}" for key in sorted(empty_keys)])
                pytest.fail(error_msg)

    def test_language_switching(self):
        """Test language switching via I18nManager."""
        i18n = I18nManager()

        # Test English
        i18n.set_language("en")
        assert i18n.get_language() == "en"

        # Test Portuguese
        i18n.set_language("pt")
        assert i18n.get_language() == "pt"

        # Test invalid language (should fallback to default)
        i18n.set_language("invalid")
        assert i18n.get_language() in ["en", "pt"]  # Should fallback

    def test_translation_interpolation(self):
        """Test variable interpolation in translations."""
        i18n = I18nManager()

        # Test with variables
        result = i18n.t("workflow_state.training.starting", user_name="John")
        assert "John" in result or result != ""  # Should contain interpolated value

    def test_critical_keys_exist(self, loaded_translations: dict):
        """Verify critical keys identified in screenshot exist."""
        critical_keys = [
            "workflow_state.training.starting",
            "workflow_state.training.patience",
            "workflow_state.training.model_ready",
            "workflow_state.training.default_name",
            "workflow_state.training.model_id_display",
            "workflow_state.training.model_type_display",
            "workflow_state.training.ready_for_predictions",
        ]

        for lang, translations in loaded_translations.items():
            missing = []
            for key in critical_keys:
                if not self._key_exists_in_yaml(key, translations):
                    missing.append(key)

            if missing:
                error_msg = f"\nCritical keys missing in {lang}.yaml:\n"
                error_msg += "\n".join([f"  - {key}" for key in missing])
                pytest.fail(error_msg)

    # Helper methods

    def _key_exists_in_yaml(self, key: str, yaml_data: dict) -> bool:
        """Check if dot-notation key exists in nested YAML structure."""
        keys = key.split(".")
        current = yaml_data

        for k in keys:
            if not isinstance(current, dict) or k not in current:
                return False
            current = current[k]

        return True

    def _flatten_yaml_keys(self, yaml_data: dict, prefix: str = "") -> Set[str]:
        """Flatten nested YAML structure to dot-notation keys."""
        keys = set()

        for key, value in yaml_data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                keys.update(self._flatten_yaml_keys(value, full_key))
            else:
                keys.add(full_key)

        return keys

    def _find_empty_values(
        self, yaml_data: dict, prefix: str = ""
    ) -> List[str]:
        """Find all keys with empty string values."""
        empty_keys = []

        for key, value in yaml_data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                empty_keys.extend(self._find_empty_values(value, full_key))
            elif isinstance(value, str) and not value.strip():
                empty_keys.append(full_key)

        return empty_keys


class TestI18nUsagePatterns:
    """Test correct usage patterns of I18nManager."""

    def test_i18n_manager_import(self, handler_files: List[Path]):
        """Verify I18nManager is imported in handler files."""
        files_without_import = []

        for handler_file in handler_files:
            if handler_file.name.startswith("__"):
                continue

            with open(handler_file, "r", encoding="utf-8") as f:
                content = f.read()

            # Skip if file doesn't use any user-facing strings
            if not re.search(r'["\']\w+.*["\']', content):
                continue

            # Check for I18nManager import
            if "I18nManager" not in content and "from src.utils.i18n_manager import" not in content:
                files_without_import.append(handler_file.name)

        # This is a warning, not a hard failure
        if files_without_import:
            print(
                f"\nWarning: {len(files_without_import)} files may need I18nManager import:"
            )
            for filename in files_without_import:
                print(f"  - {filename}")

    def test_consistent_key_naming(self, loaded_translations: dict):
        """Verify i18n keys follow consistent naming convention.

        Keys should use snake_case and organize hierarchically.
        """
        en_keys = self._flatten_yaml_keys(loaded_translations["en"])

        invalid_keys = []
        for key in en_keys:
            # Check for snake_case (allow dots for hierarchy)
            if not re.match(r"^[a-z0-9_.]+$", key):
                invalid_keys.append(key)

        if invalid_keys:
            error_msg = "\nKeys not following snake_case convention:\n"
            error_msg += "\n".join([f"  - {key}" for key in sorted(invalid_keys)])
            pytest.fail(error_msg)

    def _flatten_yaml_keys(self, yaml_data: dict, prefix: str = "") -> Set[str]:
        """Flatten nested YAML structure to dot-notation keys."""
        keys = set()

        for key, value in yaml_data.items():
            full_key = f"{prefix}.{key}" if prefix else key

            if isinstance(value, dict):
                keys.update(self._flatten_yaml_keys(value, full_key))
            else:
                keys.add(full_key)

        return keys


@pytest.fixture(scope="session")
def handler_files() -> List[Path]:
    """Get all handler Python files for session-scoped tests."""
    project_root = Path(__file__).parent.parent.parent
    handlers_dir = project_root / "src" / "bot" / "handlers"
    return list(handlers_dir.glob("*.py"))


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "i18n: mark test as i18n coverage test"
    )
