#!/usr/bin/env python3
"""Validate translation YAML files for common issues."""
import yaml
import sys
from pathlib import Path


def validate_yaml_file(yaml_path):
    """Check YAML file for boolean keys and non-string values."""
    issues = []

    print(f"Loading {yaml_path}...")
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    def check_dict(obj, path=''):
        """Recursively check dictionary for type issues."""
        if not isinstance(obj, dict):
            return

        for key, value in obj.items():
            current_path = f'{path}.{key}' if path else key

            # Check for boolean keys (YAML interprets yes/no as True/False)
            if isinstance(key, bool):
                issues.append(f"❌ Boolean key at {current_path}: {key} (should be quoted)")

            # Check for non-string keys
            elif not isinstance(key, str):
                issues.append(f"❌ Non-string key at {current_path}: {type(key).__name__} = {key}")

            # Recurse into nested dicts
            if isinstance(value, dict):
                check_dict(value, current_path)
            # Check for non-string leaf values
            elif not isinstance(value, str):
                issues.append(f"⚠️  Non-string value at {current_path}: {type(value).__name__} = {value}")

    check_dict(data)
    return issues


def main():
    """Validate both EN and PT translation files."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    all_issues = []

    for lang in ['en', 'pt']:
        yaml_file = project_root / f'locales/{lang}.yaml'

        if not yaml_file.exists():
            print(f'❌ File not found: {yaml_file}')
            sys.exit(1)

        print(f'\n{"="*60}')
        print(f'Validating {yaml_file.name}')
        print(f'{"="*60}')

        issues = validate_yaml_file(yaml_file)

        if issues:
            print(f'\n❌ Found {len(issues)} issue(s):')
            for issue in issues:
                print(f'  {issue}')
            all_issues.extend(issues)
        else:
            print('✅ No issues found')

    print(f'\n{"="*60}')
    if all_issues:
        print(f'❌ Validation FAILED: {len(all_issues)} total issue(s)')
        print(f'{"="*60}')
        sys.exit(1)
    else:
        print('✅ All translation files valid!')
        print(f'{"="*60}')
        sys.exit(0)


if __name__ == '__main__':
    main()
