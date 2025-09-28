"""
Configuration settings for data processing and validation.

This module contains configurable parameters for the data loader
and processing pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Set, List
import os


@dataclass
class DataLoaderConfig:
    """Configuration for the DataLoader class."""

    # File size limits
    max_file_size_mb: float = 10.0  # Maximum file size in MB
    min_file_size_bytes: int = 1  # Minimum file size in bytes

    # Supported file types
    supported_extensions: Set[str] = None
    supported_mime_types: Set[str] = None

    # Data validation thresholds
    max_missing_percentage: float = 90.0  # Maximum percentage of missing values
    min_rows: int = 1  # Minimum number of rows
    max_rows: int = 1_000_000  # Maximum number of rows
    max_columns: int = 1000  # Maximum number of columns

    # CSV processing settings
    csv_encodings: List[str] = None  # Encodings to try for CSV files
    csv_delimiter_detection: bool = True  # Auto-detect CSV delimiters
    csv_max_sample_size: int = 10000  # Max rows to sample for delimiter detection

    # Excel processing settings
    excel_read_first_sheet_only: bool = True  # Only read first sheet by default
    excel_max_sheets: int = 10  # Maximum number of sheets to process

    # Memory and performance
    chunk_size: int = 10000  # Chunk size for large file processing
    max_memory_mb: float = 500.0  # Maximum memory usage for data processing

    # Security settings
    allow_path_traversal: bool = False  # Allow path traversal in filenames
    sanitize_column_names: bool = True  # Clean column names automatically
    max_filename_length: int = 255  # Maximum filename length

    def __post_init__(self):
        """Initialize default values for collection fields."""
        if self.supported_extensions is None:
            self.supported_extensions = {'.csv', '.xlsx', '.xls'}

        if self.supported_mime_types is None:
            self.supported_mime_types = {
                'text/csv',
                'application/csv',
                'text/plain',  # Sometimes CSV files are detected as plain text
                'application/vnd.ms-excel',
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            }

        if self.csv_encodings is None:
            self.csv_encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return int(self.max_file_size_mb * 1024 * 1024)

    @property
    def max_memory_bytes(self) -> int:
        """Get maximum memory usage in bytes."""
        return int(self.max_memory_mb * 1024 * 1024)

    def is_supported_extension(self, extension: str) -> bool:
        """Check if file extension is supported."""
        return extension.lower() in self.supported_extensions

    def is_supported_mime_type(self, mime_type: str) -> bool:
        """Check if MIME type is supported."""
        return mime_type.lower() in self.supported_mime_types

    @classmethod
    def from_environment(cls) -> 'DataLoaderConfig':
        """Create configuration from environment variables."""
        return cls(
            max_file_size_mb=float(os.getenv('DATA_MAX_FILE_SIZE_MB', '10.0')),
            max_missing_percentage=float(os.getenv('DATA_MAX_MISSING_PCT', '90.0')),
            max_rows=int(os.getenv('DATA_MAX_ROWS', '1000000')),
            max_columns=int(os.getenv('DATA_MAX_COLUMNS', '1000')),
            chunk_size=int(os.getenv('DATA_CHUNK_SIZE', '10000')),
            max_memory_mb=float(os.getenv('DATA_MAX_MEMORY_MB', '500.0')),
            sanitize_column_names=os.getenv('DATA_SANITIZE_COLUMNS', 'true').lower() == 'true'
        )


@dataclass
class ValidationConfig:
    """Configuration for data validation rules."""

    # Column validation
    max_column_name_length: int = 100
    reserved_column_names: Set[str] = None
    allowed_column_characters: str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"

    # Data type validation
    auto_detect_types: bool = True
    numeric_threshold: float = 0.8  # Percentage of numeric values to consider column numeric
    date_detection: bool = True
    categorical_threshold: int = 50  # Max unique values for categorical detection

    # Quality checks
    check_duplicates: bool = True
    check_outliers: bool = True
    outlier_z_threshold: float = 3.0  # Z-score threshold for outlier detection

    # Sample size for validation
    validation_sample_size: int = 1000  # Rows to sample for validation checks

    def __post_init__(self):
        """Initialize default values for collection fields."""
        if self.reserved_column_names is None:
            self.reserved_column_names = {
                # SQL reserved words
                'select', 'from', 'where', 'group', 'order', 'by', 'having',
                'insert', 'update', 'delete', 'create', 'drop', 'alter',
                'table', 'index', 'view', 'database', 'schema',
                # Python reserved words
                'and', 'or', 'not', 'in', 'is', 'if', 'else', 'for', 'while',
                'def', 'class', 'import', 'from', 'as', 'try', 'except',
                'with', 'yield', 'lambda', 'global', 'nonlocal',
                # Common problematic names
                'id', 'type', 'class', 'object', 'str', 'int', 'float',
                'list', 'dict', 'set', 'tuple', 'bool'
            }

    def is_valid_column_name(self, name: str) -> bool:
        """Check if column name is valid."""
        if not name or len(name) > self.max_column_name_length:
            return False

        if name.lower() in self.reserved_column_names:
            return False

        # Check for valid characters
        return all(c in self.allowed_column_characters for c in name)


@dataclass
class SecurityConfig:
    """Security-related configuration for data processing."""

    # File validation
    validate_file_headers: bool = True  # Check file headers for type detection
    allow_compressed_files: bool = False  # Allow ZIP, GZIP files
    scan_for_macros: bool = True  # Scan Excel files for macros

    # Content validation
    max_cell_value_length: int = 10000  # Maximum length of cell values
    check_suspicious_patterns: bool = True  # Check for suspicious content patterns
    allow_formulas: bool = False  # Allow Excel formulas

    # Suspicious patterns to detect
    suspicious_patterns: List[str] = None

    # URL/path patterns
    check_for_urls: bool = True
    check_for_file_paths: bool = True

    def __post_init__(self):
        """Initialize default values for collection fields."""
        if self.suspicious_patterns is None:
            self.suspicious_patterns = [
                r'=\w+\(',  # Excel formulas
                r'javascript:',  # JavaScript URLs
                r'data:',  # Data URLs
                r'<script',  # Script tags
                r'eval\(',  # Eval functions
                r'exec\(',  # Exec functions
                r'system\(',  # System calls
                r'__import__',  # Python imports
            ]


# Global configuration instances
default_data_config = DataLoaderConfig()
default_validation_config = ValidationConfig()
default_security_config = SecurityConfig()


def get_data_config() -> DataLoaderConfig:
    """Get the current data loader configuration."""
    return default_data_config


def get_validation_config() -> ValidationConfig:
    """Get the current validation configuration."""
    return default_validation_config


def get_security_config() -> SecurityConfig:
    """Get the current security configuration."""
    return default_security_config


def update_config_from_env() -> None:
    """Update global configuration from environment variables."""
    global default_data_config
    default_data_config = DataLoaderConfig.from_environment()


# Auto-load from environment on import
if os.getenv('DATA_AUTO_CONFIG', 'true').lower() == 'true':
    update_config_from_env()