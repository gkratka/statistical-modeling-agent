"""
Script validation system for security and syntax checking.

This module provides comprehensive validation for generated scripts
to ensure they are safe to execute and syntactically correct.
"""

import ast
import re
from typing import List, Tuple, Union

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ScriptValidator:
    """Validates generated scripts for security and syntax."""

    def __init__(self) -> None:
        """Initialize the validator with security patterns."""
        self.forbidden_patterns = self._init_forbidden_patterns()
        self.allowed_imports = self._init_allowed_imports()

    def _init_forbidden_patterns(self) -> List[str]:
        """Initialize list of forbidden patterns."""
        return [
            # Code execution
            r'(__import__|exec|eval|compile)\s*\(',

            # File operations
            r'(open|file)\s*\(',
            r'with\s+open\s*\(',

            # Input operations
            r'\b(input|raw_input)\s*\(',

            # OS module operations (consolidated)
            r'os\.(system|popen|spawn|exec|fork|kill|killpg|remove|rmdir|mkdir|makedirs|chmod|chown|rename|link|symlink|environ|putenv|unsetenv)',

            # File system modules
            r'(subprocess|shutil|tempfile)\.',
            r'pathlib\.Path\s*\(',

            # Network modules (consolidated)
            r'(socket|urllib|requests|http|ftplib|telnetlib|smtplib|poplib|imaplib)\.',

            # Path traversal
            r'\.\.(/|\\|(?=\s))',

            # System introspection (consolidated)
            r'(globals|locals|vars|dir|getattr|setattr|delattr|hasattr|callable)\s*\(',
            r'(__file__|__name__.*!=.*main)',
            r'isinstance\s*\([^,]+,\s*type\s*\)',

            # Resource/threading modules (consolidated)
            r'(gc|resource|signal|threading|multiprocessing)\.',

            # Code manipulation modules (consolidated)
            r'(inspect|types|ast|dis|code)\.',

            # Exit operations (consolidated)
            r'(sys\.exit|exit|quit)\s*\(',
            r'raise\s+SystemExit',

            # Import manipulation modules (consolidated)
            r'(importlib|imp|pkgutil)\.',

            # Dangerous builtins (consolidated)
            r'(memoryview|bytearray|buffer)\s*\(',
        ]

    def _init_allowed_imports(self) -> List[str]:
        """Initialize list of allowed imports."""
        return [
            'json',
            'sys',
            'math',
            'datetime',
            'time',
            'pandas',
            'numpy',
            'scipy',
            'sklearn',
            'matplotlib',
            'seaborn',
            'plotly',
            'statistics',
            'collections',
            'itertools',
            'functools',
            'typing',
            're',
            'hashlib',
            'base64',
            'warnings',
        ]

    def validate_syntax(self, script: str) -> Tuple[bool, List[str]]:
        """
        Validate script syntax using AST parsing.

        Args:
            script: Python script to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        try:
            ast.parse(script)
            logger.debug("Script syntax validation passed")
            return True, []
        except SyntaxError as e:
            error_msg = f"Syntax error at line {e.lineno}: {e.msg}"
            logger.warning(f"Script syntax validation failed: {error_msg}")
            return False, [error_msg]
        except Exception as e:
            error_msg = f"Unexpected error during syntax validation: {str(e)}"
            logger.error(error_msg)
            return False, [error_msg]

    def check_forbidden_patterns(self, script: str) -> List[str]:
        """
        Check script for forbidden patterns.

        Args:
            script: Python script to check

        Returns:
            List of violation messages
        """
        violations = []

        for pattern in self.forbidden_patterns:
            matches = re.finditer(pattern, script, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                violation = f"Forbidden pattern '{pattern}' found: {match.group()}"
                violations.append(violation)
                logger.warning(f"Security violation detected: {violation}")

        return violations

    def validate_imports(self, ast_tree: ast.AST) -> Tuple[bool, List[str]]:
        """
        Validate that only allowed imports are used.

        Args:
            ast_tree: Parsed AST tree

        Returns:
            Tuple of (is_valid, violation_messages)
        """
        violations = []

        for node in ast.walk(ast_tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name not in self.allowed_imports:
                        violation = f"Forbidden import: {alias.name}"
                        violations.append(violation)
                        logger.warning(f"Import violation: {violation}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name not in self.allowed_imports:
                        violation = f"Forbidden import from: {node.module}"
                        violations.append(violation)
                        logger.warning(f"Import violation: {violation}")

        return len(violations) == 0, violations

    def validate_script(self, script: str) -> Tuple[bool, List[str]]:
        """
        Perform comprehensive script validation.

        Args:
            script: Python script to validate

        Returns:
            Tuple of (is_valid, all_violations)
        """
        all_violations = []

        # Check syntax first
        syntax_valid, syntax_errors = self.validate_syntax(script)
        if not syntax_valid:
            all_violations.extend(syntax_errors)
            # If syntax is invalid, we can't proceed with AST validation
            return False, all_violations

        # Check forbidden patterns
        pattern_violations = self.check_forbidden_patterns(script)
        all_violations.extend(pattern_violations)

        # Parse AST for import validation
        try:
            tree = ast.parse(script)
            import_valid, import_violations = self.validate_imports(tree)
            all_violations.extend(import_violations)
        except Exception as e:
            all_violations.append(f"AST parsing failed: {str(e)}")

        is_valid = len(all_violations) == 0

        if is_valid:
            logger.info("Script validation passed all checks")
        else:
            logger.warning(f"Script validation failed with {len(all_violations)} violations")

        return is_valid, all_violations

    def get_security_summary(self, script: str) -> dict:
        """
        Get detailed security analysis summary.

        Args:
            script: Python script to analyze

        Returns:
            Dictionary with security analysis results
        """
        is_valid, violations = self.validate_script(script)

        return {
            "is_safe": is_valid,
            "total_violations": len(violations),
            "violations": violations,
            "script_length": len(script),
            "line_count": len(script.splitlines()),
            "allowed_imports": self.allowed_imports,
            "validation_timestamp": __import__('datetime').datetime.now().isoformat()
        }