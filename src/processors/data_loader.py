"""
Data loader for handling file uploads and data processing.

This module provides secure file upload handling, validation, and
conversion to pandas DataFrames for the Statistical Modeling Agent.

Supports both:
- Telegram file uploads (original functionality)
- Local file path loading (NEW - Phase 3)
"""

import asyncio
import io
import tempfile
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging

import pandas as pd
import numpy as np
from telegram import File as TelegramFile
from telegram.ext import ContextTypes

from src.utils.exceptions import DataError, ValidationError, PathValidationError
from src.utils.logger import get_logger
from src.utils.path_validator import validate_local_path
from src.utils.schema_detector import detect_schema, DatasetSchema

logger = get_logger(__name__)


class DataLoader:
    """
    Handles file uploads from Telegram and converts them to pandas DataFrames.

    Provides secure file handling with validation, type detection,
    and data quality checks.
    """

    # Configuration constants
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_EXTENSIONS = {'.csv', '.xlsx', '.xls'}
    SUPPORTED_MIME_TYPES = {
        'text/csv',
        'application/csv',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }

    # Data validation thresholds
    MAX_MISSING_PERCENTAGE = 90.0  # Max 90% missing values
    MIN_ROWS = 1  # Minimum 1 row of data
    MAX_ROWS = 1_000_000  # Maximum 1M rows
    MAX_COLUMNS = 1000  # Maximum 1000 columns

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the data loader with optional local_data configuration."""
        self.logger = logger
        self.config = config or {}

        # Local path configuration (from config.yaml)
        local_config = self.config.get('local_data', {})
        self.local_enabled = local_config.get('enabled', False)
        self.allowed_directories = local_config.get('allowed_directories', [])
        self.local_max_size_mb = local_config.get('max_file_size_mb', 1000)
        self.local_extensions = local_config.get('allowed_extensions', ['.csv', '.xlsx', '.xls', '.parquet'])

    async def load_from_telegram(
        self,
        telegram_file: TelegramFile,
        file_name: str,
        file_size: int,
        context: ContextTypes.DEFAULT_TYPE
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Download and process a file from Telegram.

        Args:
            telegram_file: Telegram File object
            file_name: Original filename
            file_size: File size in bytes
            context: Telegram bot context

        Returns:
            Tuple of (DataFrame, metadata dict)

        Raises:
            ValidationError: If file validation fails
            DataError: If data processing fails
        """
        self.logger.info(f"Processing Telegram file: {file_name} ({file_size} bytes)")

        # Validate file before downloading
        self._validate_file_metadata(file_name, file_size)

        # Download file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(file_name)) as temp_file:
            try:
                # Download file from Telegram
                await telegram_file.download_to_drive(temp_file.name)
                temp_path = Path(temp_file.name)

                # Process the downloaded file
                df, metadata = await self._process_file(temp_path, file_name)

                self.logger.info(f"Successfully processed {file_name}: {df.shape}")
                return df, metadata

            except Exception as e:
                self.logger.error(f"Failed to process file {file_name}: {e}")
                raise DataError(
                    f"Failed to process file: {str(e)}",
                    missing_columns=[]
                ) from e
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

    async def load_from_local_path(
        self,
        file_path: str,
        detect_schema_flag: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[DatasetSchema]]:
        """
        Load data from local file path with security validation.

        Returns: (DataFrame, metadata dict, optional schema)
        Raises: PathValidationError, DataError, ValidationError
        """
        self.logger.info(f"Loading from local path: {file_path}")

        # Check if local path feature is enabled
        if not self.local_enabled:
            raise ValidationError(
                "Local file path loading is disabled. Please upload the file directly.",
                field="local_data",
                value="disabled"
            )

        # Validate path security
        is_valid, error_msg, resolved_path = validate_local_path(
            path=file_path,
            allowed_dirs=self.allowed_directories,
            max_size_mb=self.local_max_size_mb,
            allowed_extensions=self.local_extensions
        )

        if not is_valid:
            raise PathValidationError(
                message=error_msg or "Path validation failed",
                path=file_path,
                reason="security_validation"
            )

        assert resolved_path is not None, "Resolved path should not be None if validation passed"

        try:
            # Process the file (reuse existing logic)
            df, metadata = await self._process_file(resolved_path, resolved_path.name)

            # Add source information
            metadata['source'] = 'local_path'
            metadata['original_path'] = file_path
            metadata['resolved_path'] = str(resolved_path)

            # Optionally detect schema
            schema = None
            if detect_schema_flag:
                self.logger.info(f"Detecting schema for {resolved_path}")
                schema = detect_schema(resolved_path, auto_suggest=True)
                metadata['schema_detected'] = True
                metadata['suggested_task_type'] = schema.suggested_task_type
                metadata['suggested_target'] = schema.suggested_target
                metadata['suggested_features'] = schema.suggested_features
            else:
                metadata['schema_detected'] = False

            self.logger.info(f"Successfully loaded from local path: {df.shape}")
            return df, metadata, schema

        except Exception as e:
            self.logger.error(f"Failed to load from local path {file_path}: {e}")
            if isinstance(e, (DataError, ValidationError, PathValidationError)):
                raise
            else:
                raise DataError(
                    f"Failed to load file from path: {str(e)}",
                    missing_columns=[]
                ) from e

    async def _process_file(self, file_path: Path, original_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Process a file and return DataFrame with metadata.

        Args:
            file_path: Path to the file to process
            original_name: Original filename for error messages

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        extension = self._get_file_extension(original_name).lower()

        if extension == '.csv':
            return await self._read_csv(file_path, original_name)
        elif extension in {'.xlsx', '.xls'}:
            return await self._read_excel(file_path, original_name)
        else:
            raise ValidationError(
                f"Unsupported file type: {extension}",
                field="file_extension",
                value=extension
            )

    async def _read_csv(self, file_path: Path, original_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read CSV file with encoding detection and error handling.

        Args:
            file_path: Path to CSV file
            original_name: Original filename

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                # Try reading with current encoding
                df = pd.read_csv(file_path, encoding=encoding)

                # Validate the resulting DataFrame
                metadata = self._validate_dataframe(df, original_name)
                metadata['encoding'] = encoding
                metadata['file_type'] = 'csv'

                self.logger.info(f"Successfully read CSV with {encoding} encoding")
                return df, metadata

            except UnicodeDecodeError:
                self.logger.debug(f"Failed to read CSV with {encoding} encoding")
                continue
            except pd.errors.EmptyDataError:
                raise DataError(
                    "CSV file is empty or contains no data",
                    data_shape=(0, 0)
                )
            except pd.errors.ParserError as e:
                raise DataError(
                    f"CSV parsing error: {str(e)}",
                    missing_columns=[]
                )

        # If all encodings failed
        raise DataError(
            "Could not read CSV file. File may be corrupted or use an unsupported encoding.",
            missing_columns=[]
        )

    async def _read_excel(self, file_path: Path, original_name: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Read Excel file with sheet detection and error handling.

        Args:
            file_path: Path to Excel file
            original_name: Original filename

        Returns:
            Tuple of (DataFrame, metadata dict)
        """
        try:
            # Read Excel file info
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names

            # Use first sheet by default
            if not sheet_names:
                raise DataError(
                    "Excel file contains no worksheets",
                    data_shape=(0, 0)
                )

            sheet_name = sheet_names[0]
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            # Validate the resulting DataFrame
            metadata = self._validate_dataframe(df, original_name)
            metadata['file_type'] = 'excel'
            metadata['sheet_name'] = sheet_name
            metadata['available_sheets'] = sheet_names

            self.logger.info(f"Successfully read Excel sheet '{sheet_name}'")
            return df, metadata

        except FileNotFoundError:
            raise DataError(
                "Excel file not found or could not be accessed",
                missing_columns=[]
            )
        except Exception as e:
            if "xlrd" in str(e) or "openpyxl" in str(e):
                raise DataError(
                    "Excel file format not supported. Please use .xlsx format.",
                    missing_columns=[]
                )
            else:
                raise DataError(
                    f"Error reading Excel file: {str(e)}",
                    missing_columns=[]
                )

    def _validate_file_metadata(self, file_name: str, file_size: int) -> None:
        """
        Validate file metadata before processing.

        Args:
            file_name: Name of the file
            file_size: Size of the file in bytes

        Raises:
            ValidationError: If validation fails
        """
        # Check file size
        if file_size > self.MAX_FILE_SIZE:
            raise ValidationError(
                f"File too large: {file_size / 1024 / 1024:.1f}MB. "
                f"Maximum allowed: {self.MAX_FILE_SIZE / 1024 / 1024:.1f}MB",
                field="file_size",
                value=str(file_size)
            )

        if file_size == 0:
            raise ValidationError(
                "File is empty",
                field="file_size",
                value="0"
            )

        # Check file extension
        extension = self._get_file_extension(file_name).lower()
        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValidationError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {', '.join(self.SUPPORTED_EXTENSIONS)}",
                field="file_extension",
                value=extension
            )

        # Validate filename for security
        if not file_name or '..' in file_name or file_name.startswith('/'):
            raise ValidationError(
                "Invalid filename",
                field="file_name",
                value=file_name
            )

    def _validate_dataframe(self, df: pd.DataFrame, file_name: str) -> Dict[str, Any]:
        """
        Validate DataFrame and return metadata.

        Args:
            df: DataFrame to validate
            file_name: Original filename for error messages

        Returns:
            Dictionary with validation metadata

        Raises:
            DataError: If validation fails
        """
        rows, cols = df.shape

        # Check basic shape constraints
        if rows < self.MIN_ROWS:
            raise DataError(
                f"Not enough data: found {rows} rows, minimum {self.MIN_ROWS} required",
                data_shape=(rows, cols)
            )

        if rows > self.MAX_ROWS:
            raise DataError(
                f"Too much data: {rows} rows, maximum {self.MAX_ROWS} allowed",
                data_shape=(rows, cols)
            )

        if cols > self.MAX_COLUMNS:
            raise DataError(
                f"Too many columns: {cols} columns, maximum {self.MAX_COLUMNS} allowed",
                data_shape=(rows, cols)
            )

        # Check for completely empty DataFrame
        if df.empty:
            raise DataError(
                "DataFrame is empty after reading file",
                data_shape=(0, 0)
            )

        # Calculate data quality metrics
        total_cells = rows * cols
        missing_cells = df.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 100

        # Check missing data threshold
        if missing_percentage > self.MAX_MISSING_PERCENTAGE:
            raise DataError(
                f"Too much missing data: {missing_percentage:.1f}% missing, "
                f"maximum {self.MAX_MISSING_PERCENTAGE}% allowed",
                data_shape=(rows, cols)
            )

        # Clean column names
        df.columns = self._clean_column_names(df.columns)

        # Generate metadata
        metadata = {
            'shape': (rows, cols),
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_percentage': missing_percentage,
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
            'has_duplicates': df.duplicated().any(),
            'duplicate_count': df.duplicated().sum()
        }

        self.logger.info(f"DataFrame validation passed: {metadata}")
        return metadata

    def _clean_column_names(self, columns: pd.Index) -> List[str]:
        """Clean and standardize column names."""
        import re
        cleaned = []
        for i, col in enumerate(columns):
            # Convert to string, replace non-alphanumeric with underscore, handle multiple underscores
            clean = re.sub(r'[^a-zA-Z0-9_]', '_', str(col).strip())
            clean = re.sub(r'_+', '_', clean).strip('_') or f'column_{i}'

            # Handle duplicates
            original = clean
            counter = 1
            while clean in cleaned:
                clean = f"{original}_{counter}"
                counter += 1
            cleaned.append(clean)
        return cleaned

    def _get_file_extension(self, file_name: str) -> str:
        """
        Get file extension from filename.

        Args:
            file_name: Name of the file

        Returns:
            File extension including the dot
        """
        return Path(file_name).suffix.lower()

    def get_data_summary(self, df: pd.DataFrame, metadata: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the loaded data."""
        rows, cols = metadata['shape']
        file_type = metadata.get('file_type', 'csv').upper()

        # Build optional fields
        optional_fields = []
        if metadata.get('encoding'):
            optional_fields.append(f"**Encoding:** {metadata['encoding']}")
        if metadata.get('sheet_name'):
            optional_fields.append(f"**Sheet:** {metadata['sheet_name']}")

        # Column preview
        columns = metadata['columns']
        column_preview = ', '.join(columns[:5]) + ('...' if len(columns) > 5 else '')

        return f"""📊 **Data Successfully Loaded**

**Shape:** {rows:,} rows × {cols} columns
**Size:** {metadata['memory_usage_mb']:.1f} MB
**File Type:** {file_type}
{chr(10).join(optional_fields)}

**Data Quality:**
• Missing data: {metadata['missing_percentage']:.1f}%
• Duplicates: {metadata['duplicate_count']:,} rows
• Numeric columns: {len(metadata['numeric_columns'])}
• Text columns: {len(metadata['categorical_columns'])}

**Columns:** {column_preview}

✅ Ready for analysis!"""

    def get_local_path_summary(
        self,
        df: pd.DataFrame,
        metadata: Dict[str, Any],
        schema: Optional[DatasetSchema] = None
    ) -> str:
        """Generate summary for local path loaded data with schema detection."""
        rows, cols = metadata['shape']
        resolved_path = Path(metadata.get('resolved_path', ''))

        # Build base summary
        parts = [
            f"📂 **Data Loaded from Local Path**\n",
            f"**File:** `{resolved_path.name}`",
            f"**Path:** `{resolved_path.parent}`",
            f"**Shape:** {rows:,} rows × {cols} columns | {metadata['memory_usage_mb']:.1f} MB",
            f"**Type:** {metadata.get('file_type', 'csv').upper()}\n",
            f"**Quality:** {metadata['missing_percentage']:.1f}% missing | "
            f"{metadata['duplicate_count']:,} duplicates | "
            f"{len(metadata['numeric_columns'])} numeric | {len(metadata['categorical_columns'])} text\n"
        ]

        # Add schema detection if available
        if schema and metadata.get('schema_detected'):
            task_emoji = "📈" if schema.suggested_task_type == "regression" else "🎯"
            quality_pct = int(schema.overall_quality_score * 100)
            quality_emoji = "✅" if quality_pct >= 80 else "⚠️" if quality_pct >= 60 else "❌"

            task_type_display = schema.suggested_task_type.title() if schema.suggested_task_type else "Unknown"
            parts.append(
                f"**🤖 Auto-Detected:**\n"
                f"{task_emoji} **{task_type_display}** | "
                f"Target: `{schema.suggested_target}` | "
                f"{quality_emoji} {quality_pct}% quality"
            )

            if schema.suggested_features:
                feats = ', '.join(f"`{f}`" for f in schema.suggested_features[:5])
                more = f" (+{len(schema.suggested_features) - 5})" if len(schema.suggested_features) > 5 else ""
                parts.append(f"Features: {feats}{more}\n")

        # Column preview
        cols_preview = ', '.join(f"`{c}`" for c in metadata['columns'][:5])
        if len(metadata['columns']) > 5:
            cols_preview += f" (+{len(metadata['columns']) - 5})"
        parts.append(f"**Columns:** {cols_preview}\n\n✅ Ready for ML training!")

        return "\n".join(parts)


