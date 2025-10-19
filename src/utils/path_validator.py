"""Local file path validation with multi-layer security checks."""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

from src.utils.exceptions import PathValidationError


def validate_local_path(
    path: str,
    allowed_dirs: List[str],
    max_size_mb: int,
    allowed_extensions: List[str]
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
        # Layer 0: Auto-fix missing leading slash for absolute paths
        # Common user error: "Users/..." instead of "/Users/..."
        normalized_path = path
        if not path.startswith('/') and not path.startswith('./'):
            # Check for common absolute path patterns
            absolute_patterns = ['Users/', 'home/', 'var/', 'tmp/', 'opt/']
            for pattern in absolute_patterns:
                if path.startswith(pattern):
                    normalized_path = '/' + path
                    break

        # Layer 1: Path normalization (resolve symlinks, relative paths)
        try:
            resolved_path = Path(normalized_path).resolve()
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


def is_path_in_allowed_directory(path: Path, allowed_dirs: List[str]) -> bool:
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


class PathValidator:
    """
    Wrapper class for path validation functions.

    This class provides an object-oriented interface to the path validation functions.
    """

    def __init__(self, allowed_directories: List[str], max_size_mb: int, allowed_extensions: List[str]):
        """
        Initialize path validator.

        Args:
            allowed_directories: List of allowed directory paths
            max_size_mb: Maximum file size in MB
            allowed_extensions: List of allowed file extensions (e.g., ['.csv', '.xlsx'])
        """
        self.allowed_directories = allowed_directories
        self.max_size_mb = max_size_mb
        self.allowed_extensions = allowed_extensions

    def validate_path(self, path: str) -> Dict[str, Any]:
        """
        Validate a file path.

        Args:
            path: File path to validate

        Returns:
            Dictionary with 'is_valid', 'error', and 'resolved_path' keys
        """
        is_valid, error, resolved_path = validate_local_path(
            path=path,
            allowed_dirs=self.allowed_directories,
            max_size_mb=self.max_size_mb,
            allowed_extensions=self.allowed_extensions
        )

        return {
            'is_valid': is_valid,
            'error': error,
            'resolved_path': resolved_path
        }

    def validate_output_path(
        self,
        directory_path: str,
        filename: str,
        required_mb: int = 0
    ) -> Dict[str, Any]:
        """
        Validate output directory and filename for saving predictions.

        Args:
            directory_path: Directory path where file will be saved
            filename: Desired filename
            required_mb: Required disk space in MB (0 = no check)

        Returns:
            Dictionary with:
                - is_valid: bool
                - resolved_path: Path (if valid)
                - error: str (if invalid)
                - warnings: List[str] (if valid but with warnings)
        """
        import shutil

        warnings = []

        # Check for path traversal in filename BEFORE sanitization
        if detect_path_traversal(filename):
            return {
                'is_valid': False,
                'error': "Invalid filename: path traversal detected",
                'resolved_path': None,
                'warnings': []
            }

        # Sanitize filename
        sanitized_filename = self.sanitize_filename(filename)

        # Double-check after sanitization (belt and suspenders)
        if detect_path_traversal(sanitized_filename):
            return {
                'is_valid': False,
                'error': "Invalid filename: path traversal detected",
                'resolved_path': None,
                'warnings': []
            }

        # Validate directory exists
        try:
            dir_path = Path(directory_path).resolve()
        except (ValueError, OSError) as e:
            return {
                'is_valid': False,
                'error': f"Invalid directory path: {str(e)}",
                'resolved_path': None,
                'warnings': []
            }

        if not dir_path.exists():
            return {
                'is_valid': False,
                'error': f"Directory not found: {directory_path}",
                'resolved_path': None,
                'warnings': []
            }

        if not dir_path.is_dir():
            return {
                'is_valid': False,
                'error': f"Path is not a directory: {directory_path}",
                'resolved_path': None,
                'warnings': []
            }

        # Check directory is in whitelist
        if not is_path_in_allowed_directory(dir_path, self.allowed_directories):
            dirs_list = "\n".join(f"  • {d}" for d in self.allowed_directories)
            return {
                'is_valid': False,
                'error': f"Directory not in allowed paths.\n\nAllowed:\n{dirs_list}",
                'resolved_path': None,
                'warnings': []
            }

        # Check directory is writable
        if not os.access(dir_path, os.W_OK):
            return {
                'is_valid': False,
                'error': f"Directory is not writable (permission denied): {directory_path}",
                'resolved_path': None,
                'warnings': []
            }

        # Validate file extension
        file_ext = Path(sanitized_filename).suffix.lower()
        if file_ext not in [ext.lower() for ext in self.allowed_extensions]:
            # Auto-add .csv if no extension
            if not file_ext:
                sanitized_filename += '.csv'
            else:
                exts_list = ", ".join(self.allowed_extensions)
                return {
                    'is_valid': False,
                    'error': f"Invalid file extension: {file_ext}\nAllowed: {exts_list}",
                    'resolved_path': None,
                    'warnings': []
                }

        # Build full path
        full_path = dir_path / sanitized_filename

        # Check if file already exists
        if full_path.exists():
            warnings.append(f"File already exists: {full_path.name}")

        # Check disk space if required
        if required_mb > 0:
            if not self.check_disk_space(dir_path, required_mb):
                usage = shutil.disk_usage(dir_path)
                free_mb = usage.free / (1024 * 1024)
                return {
                    'is_valid': False,
                    'error': f"Insufficient disk space. Required: {required_mb}MB, Available: {free_mb:.1f}MB",
                    'resolved_path': None,
                    'warnings': []
                }

        return {
            'is_valid': True,
            'error': None,
            'resolved_path': full_path,
            'warnings': warnings
        }

    def sanitize_filename(self, filename: str) -> str:
        """
        Remove/replace invalid filename characters.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for filesystem
        """
        import re

        # Define invalid characters for filenames
        # Windows: < > : " / \ | ? *
        # Unix: /
        invalid_chars = r'[<>:"/\\|?*]'

        # Replace invalid characters with underscore
        sanitized = re.sub(invalid_chars, '_', filename)

        # Remove leading/trailing spaces and dots
        sanitized = sanitized.strip(' .')

        # Ensure not empty
        if not sanitized:
            sanitized = "output"

        return sanitized

    def check_disk_space(self, path: Path, required_mb: int) -> bool:
        """
        Verify sufficient disk space available.

        Args:
            path: Directory path to check
            required_mb: Required space in megabytes

        Returns:
            True if sufficient space available, False otherwise
        """
        import shutil

        try:
            usage = shutil.disk_usage(path)
            free_mb = usage.free / (1024 * 1024)
            return free_mb >= required_mb
        except OSError:
            # If we can't check, assume insufficient space (fail safe)
            return False
