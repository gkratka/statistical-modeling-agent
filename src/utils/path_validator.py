"""Local file path validation with multi-layer security checks."""

import os
from pathlib import Path
from typing import Optional, Tuple

from src.utils.exceptions import PathValidationError


def validate_local_path(
    path: str,
    allowed_dirs: list[str],
    max_size_mb: int,
    allowed_extensions: list[str]
) -> Tuple[bool, Optional[str], Optional[Path]]:
    """
    Comprehensive local file path validation with multi-layer security.

    Args:
        path: User-provided file path (can be relative or absolute)
        allowed_dirs: List of allowed directory paths (must be absolute)
        max_size_mb: Maximum allowed file size in megabytes
        allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.xlsx'])

    Returns:
        Tuple of (is_valid, error_message, resolved_path)

    Security layers: path normalization, whitelist enforcement, traversal detection,
    extension validation, existence/readability checks, size validation.
    """
    try:
        # Layer 1: Path normalization (resolve symlinks, relative paths)
        try:
            resolved_path = Path(path).resolve()
        except (ValueError, OSError) as e:
            return False, f"Invalid path format: {str(e)}", None

        # Layer 2: Path traversal detection (early check before other operations)
        if detect_path_traversal(path):
            return (
                False,
                "Path traversal detected: Path contains suspicious patterns (../, ..\\)",
                None
            )

        # Layer 3: Directory whitelist enforcement
        if not is_path_in_allowed_directory(resolved_path, allowed_dirs):
            dirs_list = "\n".join(f"  • {d}" for d in allowed_dirs)
            return (
                False,
                f"Path not in allowed directories.\n\nAllowed:\n{dirs_list}",
                None
            )

        # Layer 4: File existence and type checks (before extension to give better error messages)
        # Layer 4a: Explicit directory check (catch directory paths early with clear message)
        if resolved_path.exists() and resolved_path.is_dir():
            return (
                False,
                f"❌ Path is a directory, not a file: {resolved_path}\n\n"
                f"Please provide the full path to a data file.",
                None
            )

        # Layer 4b: File existence check
        if not resolved_path.exists():
            return False, f"❌ File not found: {resolved_path}", None

        # Layer 4c: Special file check (not regular file)
        if not resolved_path.is_file():
            return False, f"❌ Not a regular file (special file or symlink issue): {resolved_path}", None

        # Layer 5: Extension validation (case-insensitive)
        if resolved_path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
            exts_list = ", ".join(allowed_extensions)
            return (
                False,
                f"Invalid file extension: {resolved_path.suffix}\n"
                f"Allowed: {exts_list}",
                None
            )

        # Layer 6: File readability check
        if not os.access(resolved_path, os.R_OK):
            return (
                False,
                f"File not readable (permission denied): {resolved_path}",
                None
            )

        # Layer 7: File size validation
        size_mb = get_file_size_mb(resolved_path)
        if size_mb > max_size_mb:
            return (
                False,
                f"File too large: {size_mb:.1f}MB (maximum: {max_size_mb}MB)",
                None
            )

        # Layer 8: Zero-byte file check
        if size_mb == 0:
            return False, "File is empty (0 bytes)", None

        # All checks passed
        return True, None, resolved_path

    except Exception as e:
        # Catch-all for unexpected errors
        return False, f"Validation error: {str(e)}", None


def is_path_in_allowed_directory(path: Path, allowed_dirs: list[str]) -> bool:
    """
    Check if resolved path is within any of the allowed directories.

    Uses string prefix matching on resolved absolute paths to ensure
    the file is within an allowed directory tree.
    """
    # Resolve the path to absolute (handles symlinks like /var -> /private/var on macOS)
    try:
        resolved_path = path.resolve()
        path_str = str(resolved_path)
    except (ValueError, OSError):
        # If path can't be resolved, use it as-is
        path_str = str(path)

    for allowed_dir in allowed_dirs:
        try:
            # Resolve allowed directory to absolute path (even if doesn't exist)
            allowed_path = Path(allowed_dir).expanduser().resolve()
            allowed_str = str(allowed_path)

            # Check if path starts with allowed directory
            # Use os.path.commonpath for more robust checking
            try:
                # Check if they share common path and file is under allowed dir
                common = os.path.commonpath([path_str, allowed_str])
                if common == allowed_str:
                    return True
            except ValueError:
                # Different drives on Windows or other path incompatibilities
                # Fall back to string prefix matching
                if path_str.startswith(allowed_str + os.sep) or path_str == allowed_str:
                    return True
        except (ValueError, OSError):
            # Skip invalid allowed_dir entries
            continue

    return False


def detect_path_traversal(path: str) -> bool:
    """
    Detect path traversal attempts in file paths.

    Checks for common path traversal patterns:
    - ../ (Unix), ..\\ (Windows)
    - URL-encoded variants: %2e%2e, ..%2f, ..%5c, etc.
    """
    dangerous_patterns = [
        '../',      # Unix path traversal
        '..\\',     # Windows path traversal
        '%2e%2e',   # URL-encoded ..
        '..%2f',    # Mixed encoding (Unix)
        '..%5c',    # Mixed encoding (Windows)
        '%2e%2e%2f', # Fully encoded ../
        '%2e%2e%5c', # Fully encoded ..\\
    ]

    path_lower = path.lower()
    return any(pattern in path_lower for pattern in dangerous_patterns)


def get_file_size_mb(path: Path) -> float:
    """Get file size in megabytes."""
    try:
        size_bytes = path.stat().st_size
        return size_bytes / (1024 * 1024)
    except OSError as e:
        raise PathValidationError(
            f"Cannot access file size: {str(e)}",
            path=str(path),
            reason="stat_failed"
        )
