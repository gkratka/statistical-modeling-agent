# Local File Path Training Workflow - Implementation Plan

**Status**: 🟡 Planning Complete, Ready for Implementation
**Created**: 2025-10-06
**Estimated Duration**: 10-14 hours
**Priority**: HIGH

---

## 📋 Executive Summary

### Problem Statement
Current ML training workflow only supports Telegram file uploads, which has significant limitations:
- ❌ File size limits (~50MB Telegram restriction)
- ❌ Privacy concerns (data passes through Telegram servers)
- ❌ Inefficient for large datasets (>100MB)
- ❌ Cannot handle datasets that don't fit in Telegram

### Proposed Solution
Implement **local file path training workflow** allowing users to:
- ✅ Provide local filesystem paths to datasets
- ✅ Train on large files (up to 1GB, configurable)
- ✅ Keep sensitive data private (never sent to Telegram)
- ✅ Use same familiar workflow as Telegram uploads

### Key Features
1. **Security-First**: Multi-layer path validation with directory whitelisting
2. **Auto-Schema Detection**: Intelligent type inference, minimal user input
3. **Backward Compatible**: Existing Telegram upload workflow unchanged
4. **User-Friendly**: Clear prompts, helpful errors, progress indicators
5. **Feature-Flagged**: Can be enabled/disabled via configuration

---

## 🏗️ Architecture Design

### Enhanced Workflow State Machine

```
/train command
    ↓
📂 AWAITING_DATA_SOURCE (NEW)
    ├─ "1. Upload via Telegram" → AWAITING_DATA (existing path)
    └─ "2. Provide local path" → AWAITING_FILE_PATH (NEW)
                                      ↓
                                 🔍 Auto-validation + schema detection
                                      ↓
                                 📋 CONFIRMING_SCHEMA (NEW - show detected schema)
                                      ├─ "Looks good" → SELECTING_TARGET
                                      └─ "Needs correction" → AWAITING_FILE_PATH (retry)
                                                                  ↓
                                                             SELECTING_TARGET (existing)
                                                                  ↓
                                                             SELECTING_FEATURES
                                                                  ↓
                                                             CONFIRMING_MODEL
                                                                  ↓
                                                             [Continue normal workflow...]
```

### New Components

**1. Path Validator** (`src/utils/path_validator.py`)
- Multi-layer security validation
- Directory whitelist enforcement
- Path traversal prevention
- File size/type validation

**2. Schema Detector** (`src/processors/schema_detector.py`)
- Auto-detect column types (numeric, text, category, datetime)
- Calculate statistics (missing %, unique counts)
- Format user-friendly schema display
- Extract sample values for verification

**3. Data Loader Enhancement** (`src/processors/data_loader.py`)
- Support LOCAL_FILE_PATH data source
- Chunked reading for large files
- Handle CSV, Excel, Parquet formats
- Encoding detection

**4. Workflow Handlers** (`src/bot/workflow_handlers.py`)
- `handle_data_source_selection()` - Choose Telegram vs Local
- `handle_file_path_input()` - Collect and validate path
- `handle_schema_confirmation()` - Display schema, get approval

**5. State Management** (`src/core/state_manager.py`)
- Add 4 new MLTrainingState values
- Update state transitions
- Maintain session context

---

## 🔒 Security Architecture

### 🔴 CRITICAL: Multi-Layer Security Validation

```python
# Layer 1: Path Validation
✅ Normalize path (resolve symlinks, relative paths)
✅ Check against whitelist directories
✅ Prevent path traversal (../, ..\\)
✅ Validate file extension (.csv, .xlsx, .parquet only)

# Layer 2: File Validation
✅ Check file exists and readable
✅ Verify file size within limits (configurable max)
✅ Validate file permissions
✅ Check for malicious file patterns

# Layer 3: Content Validation
✅ Verify file is valid CSV/Excel/Parquet (not disguised)
✅ Check encoding (UTF-8, Latin-1)
✅ Detect data corruption
✅ Validate column structure
```

### Configuration Schema

```yaml
# config/config.yaml (NEW SECTION)
local_data:
  enabled: true  # Feature flag
  allowed_directories:
    - "/Users/gkratka/Documents/datasets"
    - "/Users/gkratka/Documents/statistical-modeling-agent/data"
  max_file_size_mb: 1000  # 1GB limit
  allowed_extensions:
    - ".csv"
    - ".xlsx"
    - ".xls"
    - ".parquet"
  require_explicit_approval: false  # Future: Ask before accessing file
```

### Security Test Coverage Requirements

**100% Coverage for Path Validation** (15+ attack scenarios):
- ✅ Valid paths in allowed directories
- ❌ Path traversal: `../../etc/passwd`
- ❌ Paths outside whitelist
- ❌ Non-existent files
- ❌ Oversized files
- ❌ Invalid extensions
- ❌ Symlink attacks pointing outside whitelist
- ❌ Permission denied scenarios
- ❌ Relative paths (must normalize first)
- ❌ Special characters in paths
- ❌ Hidden files/directories
- ❌ Case sensitivity tricks
- ❌ Unicode/encoding attacks
- ❌ Zero-byte files
- ❌ Corrupted file headers

---

## 📝 Implementation Phases

### **Phase 1: Security Foundation** 🔴 CRITICAL
**Duration**: 1-2 hours | **Priority**: HIGHEST

#### Files to Create
1. `config/config.yaml` - Add `local_data` section
2. `src/utils/path_validator.py` (~200 lines)
3. `tests/unit/test_path_validator.py` (~300 lines, 15+ tests)

#### Implementation Details

**Path Validator** (`src/utils/path_validator.py`):
```python
class PathValidationError(ValidationError):
    """Local file path validation failures."""
    pass

def validate_local_path(
    path: str,
    allowed_dirs: list[str],
    max_size_mb: int,
    allowed_extensions: list[str]
) -> tuple[bool, Optional[str], Optional[Path]]:
    """
    Comprehensive local file path validation.

    Returns:
        (is_valid, error_message, resolved_path)

    Security Checks:
    1. Path normalization (resolve symlinks, relative paths)
    2. Directory whitelist enforcement
    3. Path traversal prevention
    4. Extension validation
    5. File existence and readability
    6. File size validation
    """
    try:
        # 1. Normalize and resolve path
        resolved_path = Path(path).resolve()

        # 2. Check whitelist
        if not is_path_in_allowed_directory(resolved_path, allowed_dirs):
            return False, f"Path not in allowed directories", None

        # 3. Detect path traversal
        if detect_path_traversal(path):
            return False, "Path traversal detected", None

        # 4. Validate extension
        if resolved_path.suffix.lower() not in allowed_extensions:
            return False, f"Invalid extension: {resolved_path.suffix}", None

        # 5. Check file exists and readable
        if not resolved_path.exists():
            return False, f"File not found: {resolved_path}", None

        if not resolved_path.is_file():
            return False, f"Not a file: {resolved_path}", None

        if not os.access(resolved_path, os.R_OK):
            return False, f"File not readable: {resolved_path}", None

        # 6. Check file size
        size_mb = get_file_size_mb(resolved_path)
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.1f}MB (max: {max_size_mb}MB)", None

        return True, None, resolved_path

    except Exception as e:
        return False, f"Validation error: {str(e)}", None

def is_path_in_allowed_directory(path: Path, allowed_dirs: list[str]) -> bool:
    """Check if resolved path is within allowed directories."""
    path_str = str(path)
    for allowed_dir in allowed_dirs:
        allowed_path = Path(allowed_dir).resolve()
        if path_str.startswith(str(allowed_path)):
            return True
    return False

def detect_path_traversal(path: str) -> bool:
    """Detect path traversal attempts."""
    dangerous_patterns = ['../', '..\\', '%2e%2e', '..%2f', '..%5c']
    return any(pattern in path.lower() for pattern in dangerous_patterns)

def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    return path.stat().st_size / (1024 * 1024)
```

#### Test Cases (`tests/unit/test_path_validator.py`)
```python
class TestPathValidator:
    """Test path validation security."""

    def test_valid_path_in_whitelist(self):
        """Valid path in allowed directory should pass."""

    def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked."""

    def test_path_outside_whitelist_blocked(self):
        """Paths outside whitelist should be blocked."""

    def test_file_not_found_rejected(self):
        """Non-existent files should be rejected."""

    def test_oversized_file_rejected(self):
        """Files exceeding size limit should be rejected."""

    def test_invalid_extension_rejected(self):
        """Invalid file extensions should be rejected."""

    def test_symlink_outside_whitelist_blocked(self):
        """Symlinks pointing outside whitelist should be blocked."""

    def test_permission_denied_handled(self):
        """Permission errors should be handled gracefully."""

    def test_relative_path_normalized(self):
        """Relative paths should be normalized before validation."""

    def test_directory_rejected(self):
        """Directories should be rejected (files only)."""

    def test_hidden_file_handled(self):
        """Hidden files should be validated same as normal files."""

    def test_unicode_path_handled(self):
        """Unicode characters in paths should be handled."""

    def test_zero_byte_file_rejected(self):
        """Empty files should be rejected."""

    def test_multiple_extensions_handled(self):
        """Files like data.csv.gz should validate on final extension."""

    def test_case_insensitive_extension(self):
        """Extensions should be case-insensitive (.CSV == .csv)."""
```

#### Acceptance Criteria
- ✅ All 15+ security tests passing
- ✅ Zero path traversal vulnerabilities
- ✅ Strict whitelist enforcement
- ✅ Clear error messages for all failure cases
- ✅ 100% test coverage for path_validator.py

---

### **Phase 2: Schema Detection & Display** 🔴 HIGH
**Duration**: 1-2 hours | **Priority**: HIGH

#### Files to Create
1. `src/processors/schema_detector.py` (~250 lines)
2. `tests/unit/test_schema_detector.py` (~200 lines, 10+ tests)

#### Implementation Details

**Schema Detector** (`src/processors/schema_detector.py`):
```python
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import pandas as pd

@dataclass
class ColumnSchema:
    """Schema information for a single column."""
    name: str
    dtype: str  # "numeric", "text", "category", "datetime", "boolean"
    nullable: bool
    unique_count: int
    sample_values: list[Any]
    missing_count: int
    missing_percentage: float

@dataclass
class DatasetSchema:
    """Complete dataset schema information."""
    file_path: Path
    columns: list[ColumnSchema]
    total_rows: int
    total_columns: int
    memory_usage_mb: float
    file_size_mb: float

def detect_schema(
    file_path: Path,
    sample_rows: int = 1000
) -> DatasetSchema:
    """
    Auto-detect schema from file with intelligent type inference.

    Detection Strategy:
    1. Read first N rows for quick analysis
    2. Infer dtypes with pandas (numeric, object, datetime)
    3. Detect categories (low cardinality text columns)
    4. Calculate statistics (missing %, unique counts)
    5. Extract sample values for user verification

    Args:
        file_path: Path to dataset file
        sample_rows: Number of rows to sample for analysis

    Returns:
        DatasetSchema with complete column information
    """
    # Load file based on extension
    ext = file_path.suffix.lower()
    if ext == '.csv':
        df = pd.read_csv(file_path, nrows=sample_rows)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path, nrows=sample_rows)
    elif ext == '.parquet':
        df = pd.read_parquet(file_path)[:sample_rows]
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # Analyze each column
    columns = []
    for col in df.columns:
        col_data = df[col]

        # Infer dtype
        dtype = infer_column_type(col_data)

        # Calculate statistics
        missing_count = col_data.isna().sum()
        missing_pct = (missing_count / len(col_data)) * 100
        unique_count = col_data.nunique()

        # Extract sample values (non-null)
        sample_values = col_data.dropna().head(3).tolist()

        columns.append(ColumnSchema(
            name=col,
            dtype=dtype,
            nullable=missing_count > 0,
            unique_count=unique_count,
            sample_values=sample_values,
            missing_count=missing_count,
            missing_percentage=missing_pct
        ))

    # Get file size
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

    return DatasetSchema(
        file_path=file_path,
        columns=columns,
        total_rows=len(df),
        total_columns=len(df.columns),
        memory_usage_mb=memory_mb,
        file_size_mb=file_size_mb
    )

def infer_column_type(series: pd.Series) -> str:
    """
    Infer high-level column type from pandas Series.

    Returns: "numeric", "text", "category", "datetime", "boolean"
    """
    # Check for numeric
    if pd.api.types.is_numeric_dtype(series):
        # Check for boolean (0/1 only)
        if series.dropna().isin([0, 1]).all():
            return "boolean"
        return "numeric"

    # Check for datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Check for category (low cardinality)
    unique_ratio = series.nunique() / len(series)
    if unique_ratio < 0.05:  # Less than 5% unique values
        return "category"

    return "text"

def format_schema_for_telegram(schema: DatasetSchema) -> str:
    """
    Format schema as user-friendly Telegram message.

    Example Output:
    📊 Dataset Schema Detected

    📁 File: german_credit_data.csv
    📐 Shape: 799 rows × 21 columns
    💾 Size: 63.3 KB

    📋 Columns:
    1. ✅ age (numeric) - 799 values, 0% missing
       Sample: [25, 30, 35]
    2. ✅ income (numeric) - 799 values, 0% missing
       Sample: [30000, 45000, 55000]
    3. ✅ class (category: 2 unique) - 799 values, 0% missing
       Sample: [0, 1]
    ...

    ✅ Ready for training? Reply 'yes' to continue.
    ⚠️ Wrong columns? Reply 'retry' to provide new path.
    """
    lines = [
        "📊 **Dataset Schema Detected**\n",
        f"📁 **File**: `{schema.file_path.name}`",
        f"📐 **Shape**: {schema.total_rows:,} rows × {schema.total_columns} columns",
        f"💾 **Size**: {schema.file_size_mb:.1f} MB\n",
        "📋 **Columns**:"
    ]

    # Show first 10 columns (or all if less than 10)
    display_cols = schema.columns[:10]
    for i, col in enumerate(display_cols, 1):
        # Status indicator
        if col.missing_percentage > 50:
            indicator = "⚠️"
        elif col.missing_percentage > 0:
            indicator = "⚡"
        else:
            indicator = "✅"

        # Format column info
        type_info = col.dtype
        if col.dtype == "category":
            type_info = f"{col.dtype}: {col.unique_count} unique"

        lines.append(
            f"{i}. {indicator} **{col.name}** ({type_info})\n"
            f"   • {schema.total_rows - col.missing_count:,} values, "
            f"{col.missing_percentage:.1f}% missing\n"
            f"   • Sample: {col.sample_values}"
        )

    if schema.total_columns > 10:
        lines.append(f"\n...and {schema.total_columns - 10} more columns")

    lines.extend([
        "\n✅ **Ready for training?** Reply **'yes'** to continue.",
        "⚠️ **Wrong file?** Reply **'retry'** to provide new path.",
        "❌ **Cancel?** Type **/cancel** to cancel workflow."
    ])

    return "\n".join(lines)
```

#### Test Cases (`tests/unit/test_schema_detector.py`)
```python
class TestSchemaDetector:
    """Test schema detection functionality."""

    def test_detect_numeric_columns(self):
        """Numeric columns should be detected correctly."""

    def test_detect_text_columns(self):
        """Text columns should be detected correctly."""

    def test_detect_category_columns(self):
        """Low-cardinality columns should be detected as categories."""

    def test_detect_datetime_columns(self):
        """Datetime columns should be detected correctly."""

    def test_detect_boolean_columns(self):
        """Binary 0/1 columns should be detected as boolean."""

    def test_missing_values_calculated(self):
        """Missing value statistics should be accurate."""

    def test_sample_values_extracted(self):
        """Sample values should be extracted correctly."""

    def test_csv_file_schema_detection(self):
        """CSV files should be processed correctly."""

    def test_excel_file_schema_detection(self):
        """Excel files should be processed correctly."""

    def test_large_file_sampled(self):
        """Large files should be sampled, not fully loaded."""

    def test_schema_formatting_telegram(self):
        """Telegram formatting should be readable and complete."""
```

#### Acceptance Criteria
- ✅ Accurate type inference (>95% accuracy on test datasets)
- ✅ Handles CSV, Excel, Parquet files
- ✅ Efficient sampling for large files
- ✅ Clear Telegram formatting with emojis/structure
- ✅ Missing value statistics accurate
- ✅ Sample values extracted correctly

---

### **Phase 3: Data Loader Enhancement** 🔴 HIGH
**Duration**: 1 hour | **Priority**: HIGH

#### Files to Modify
1. `src/processors/data_loader.py` (extend existing)
2. `tests/unit/test_data_loader_local.py` (NEW - ~150 lines)

#### Implementation Details

**Data Loader Enhancement** (`src/processors/data_loader.py`):
```python
from enum import Enum
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import pandas as pd

class DataSource(Enum):
    """Data source types."""
    TELEGRAM_UPLOAD = "telegram_upload"
    LOCAL_FILE_PATH = "local_file_path"  # NEW

@dataclass
class DataLoadConfig:
    """Configuration for data loading."""
    source: DataSource
    file_path: Optional[Path] = None  # For LOCAL_FILE_PATH
    telegram_file_id: Optional[str] = None  # For TELEGRAM_UPLOAD
    chunk_size: int = 10000  # For large files

def load_data(
    config: DataLoadConfig,
    user_id: int,
    allowed_dirs: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Load data from Telegram upload or local path.

    Routing:
    - TELEGRAM_UPLOAD → existing load_telegram_file()
    - LOCAL_FILE_PATH → new load_local_file()

    Args:
        config: Data loading configuration
        user_id: User ID for logging/tracking
        allowed_dirs: Allowed directories (for LOCAL_FILE_PATH validation)

    Returns:
        Loaded DataFrame

    Raises:
        DataError: If loading fails
    """
    if config.source == DataSource.TELEGRAM_UPLOAD:
        return load_telegram_file(config.telegram_file_id, user_id)
    elif config.source == DataSource.LOCAL_FILE_PATH:
        return load_local_file(config.file_path, config.chunk_size)
    else:
        raise ValueError(f"Unknown data source: {config.source}")

def load_local_file(
    file_path: Path,
    chunk_size: int = 10000
) -> pd.DataFrame:
    """
    Load data from validated local file path.

    Features:
    - Chunked reading for large files (>100MB)
    - Auto-detect file type (CSV, Excel, Parquet)
    - Encoding detection (UTF-8, Latin-1, etc.)
    - Memory-efficient loading

    Args:
        file_path: Validated path to data file
        chunk_size: Rows per chunk for large files

    Returns:
        Loaded DataFrame

    Raises:
        DataError: If file cannot be loaded
    """
    try:
        ext = file_path.suffix.lower()
        file_size_mb = file_path.stat().st_size / (1024 * 1024)

        # Strategy: Use chunking for files >100MB
        use_chunking = file_size_mb > 100

        if ext == '.csv':
            if use_chunking:
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                    chunks.append(chunk)
                return pd.concat(chunks, ignore_index=True)
            else:
                return pd.read_csv(file_path)

        elif ext in ['.xlsx', '.xls']:
            # Excel doesn't support chunking well
            return pd.read_excel(file_path)

        elif ext == '.parquet':
            # Parquet is already efficient
            return pd.read_parquet(file_path)

        else:
            raise DataError(f"Unsupported file type: {ext}")

    except Exception as e:
        raise DataError(f"Failed to load file {file_path}: {str(e)}")
```

#### Test Cases (`tests/unit/test_data_loader_local.py`)
```python
class TestLocalFileLoading:
    """Test local file path loading."""

    def test_load_csv_small(self):
        """Small CSV files should load completely."""

    def test_load_csv_large_chunked(self):
        """Large CSV files should use chunked loading."""

    def test_load_excel_file(self):
        """Excel files should load correctly."""

    def test_load_parquet_file(self):
        """Parquet files should load efficiently."""

    def test_encoding_detection(self):
        """Non-UTF8 encodings should be detected."""

    def test_corrupted_file_error(self):
        """Corrupted files should raise DataError."""

    def test_empty_file_error(self):
        """Empty files should raise DataError."""

    def test_unsupported_extension_error(self):
        """Unsupported extensions should raise DataError."""
```

#### Acceptance Criteria
- ✅ Loads CSV, Excel, Parquet files
- ✅ Chunked loading for files >100MB
- ✅ Encoding detection handles common encodings
- ✅ Memory-efficient for large datasets
- ✅ Clear error messages on failure

---

### **Phase 4: State Management Updates** 🟡 MEDIUM
**Duration**: 30 minutes | **Priority**: MEDIUM

#### Files to Modify
1. `src/core/state_manager.py` (extend existing MLTrainingState enum)
2. `tests/unit/test_state_manager.py` (add 5+ tests)

#### Implementation Details

```python
class MLTrainingState(str, Enum):
    """ML training workflow states."""
    IDLE = "idle"

    # NEW: Local path workflow states
    AWAITING_DATA_SOURCE = "awaiting_data_source"  # Choose Telegram vs Local
    AWAITING_FILE_PATH = "awaiting_file_path"      # Collect path
    VALIDATING_FILE_PATH = "validating_file_path"  # Auto-validation
    CONFIRMING_SCHEMA = "confirming_schema"        # Show detected schema

    # Existing states
    AWAITING_DATA = "awaiting_data"  # Telegram upload
    DATA_LOADED = "data_loaded"
    SELECTING_TARGET = "selecting_target"
    SELECTING_FEATURES = "selecting_features"
    CONFIRMING_MODEL = "confirming_model"
    SPECIFYING_ARCHITECTURE = "specifying_architecture"
    COLLECTING_HYPERPARAMETERS = "collecting_hyperparameters"
    TRAINING = "training"
    COMPLETED = "completed"
    ERROR = "error"

# State transitions (update existing)
STATE_TRANSITIONS = {
    IDLE: [AWAITING_DATA_SOURCE],  # Modified: was AWAITING_DATA
    AWAITING_DATA_SOURCE: [AWAITING_DATA, AWAITING_FILE_PATH],  # NEW
    AWAITING_FILE_PATH: [VALIDATING_FILE_PATH, AWAITING_DATA_SOURCE],  # NEW
    VALIDATING_FILE_PATH: [CONFIRMING_SCHEMA, AWAITING_FILE_PATH],  # NEW
    CONFIRMING_SCHEMA: [SELECTING_TARGET, AWAITING_FILE_PATH],  # NEW
    AWAITING_DATA: [DATA_LOADED],  # Existing Telegram path
    DATA_LOADED: [SELECTING_TARGET],  # Existing
    # ... rest of existing transitions
}
```

#### Test Cases
```python
def test_new_states_added():
    """New states should be in MLTrainingState enum."""

def test_awaiting_data_source_transitions():
    """AWAITING_DATA_SOURCE should transition to AWAITING_DATA or AWAITING_FILE_PATH."""

def test_local_path_workflow_transitions():
    """Local path workflow should follow correct state sequence."""

def test_telegram_workflow_unchanged():
    """Telegram workflow should still work (backward compatibility)."""

def test_invalid_transitions_blocked():
    """Invalid state transitions should be blocked."""
```

#### Acceptance Criteria
- ✅ 4 new states added to MLTrainingState
- ✅ State transitions updated correctly
- ✅ Backward compatibility maintained
- ✅ All state tests passing

---

### **Phase 5: Workflow Handlers Implementation** 🔴 HIGH
**Duration**: 2-3 hours | **Priority**: HIGH

#### Files to Modify
1. `src/bot/workflow_handlers.py` (add new handlers)
2. `src/bot/handlers.py` (modify `/train` command)
3. `tests/integration/test_local_path_workflow.py` (NEW - ~400 lines)

#### Implementation Details

**New Handler Functions** (`src/bot/workflow_handlers.py`):

```python
async def handle_data_source_selection(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: ConversationSession
) -> None:
    """
    Handle data source selection (Step 1 of workflow).

    User Input: "1" (Telegram) or "2" (Local path)

    State Transitions:
    - "1" → AWAITING_DATA (existing Telegram upload flow)
    - "2" → AWAITING_FILE_PATH (new local path flow)
    """
    user_input = update.message.text.strip()

    if user_input == "1":
        # Telegram upload path (existing)
        session.workflow_state = MLTrainingState.AWAITING_DATA
        message = (
            "📤 **Upload Dataset**\n\n"
            "Please upload your dataset file.\n\n"
            "**Supported formats:** CSV, Excel, Parquet\n"
            "**Max size:** 50 MB\n\n"
            "Type **/cancel** to cancel workflow."
        )

    elif user_input == "2":
        # Local path flow (NEW)
        config = context.bot_data.get("config")
        allowed_dirs = config.get("local_data", {}).get("allowed_directories", [])

        session.workflow_state = MLTrainingState.AWAITING_FILE_PATH
        message = FILE_PATH_PROMPT.format(
            allowed_dirs="\n".join(f"• `{d}`" for d in allowed_dirs)
        )

    else:
        # Invalid input
        message = (
            "⚠️ Invalid choice. Please reply with **1** or **2**.\n\n"
            "1️⃣ Upload via Telegram\n"
            "2️⃣ Provide local file path"
        )
        # Stay in same state
        return

    await update.message.reply_text(message, parse_mode="Markdown")

async def handle_file_path_input(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: ConversationSession
) -> None:
    """
    Handle local file path input and validation.

    User Input: "/Users/username/datasets/data.csv"

    Process:
    1. Validate path with path_validator
    2. If valid → Detect schema → CONFIRMING_SCHEMA
    3. If invalid → Show error, retry AWAITING_FILE_PATH
    """
    file_path_str = update.message.text.strip()

    # Get config
    config = context.bot_data.get("config")
    allowed_dirs = config.get("local_data", {}).get("allowed_directories", [])
    max_size_mb = config.get("local_data", {}).get("max_file_size_mb", 1000)
    allowed_exts = config.get("local_data", {}).get("allowed_extensions", [".csv", ".xlsx", ".parquet"])

    # Validate path
    is_valid, error_msg, resolved_path = validate_local_path(
        file_path_str,
        allowed_dirs,
        max_size_mb,
        allowed_exts
    )

    if not is_valid:
        # Show error, stay in AWAITING_FILE_PATH
        error_response = format_path_error(error_msg, allowed_dirs, max_size_mb, allowed_exts)
        await update.message.reply_text(error_response, parse_mode="Markdown")
        return

    # Path valid! Detect schema
    try:
        schema = detect_schema(resolved_path)
        session.selections["file_path"] = str(resolved_path)
        session.selections["schema"] = schema

        # Format and display schema
        schema_message = format_schema_for_telegram(schema)

        session.workflow_state = MLTrainingState.CONFIRMING_SCHEMA
        await update.message.reply_text(schema_message, parse_mode="Markdown")

    except Exception as e:
        error_message = (
            f"❌ **Failed to load file**\n\n"
            f"Error: {str(e)}\n\n"
            f"Please check:\n"
            f"• File is not corrupted\n"
            f"• File format is valid\n"
            f"• You have read permissions\n\n"
            f"Provide a different path or type **/cancel** to cancel."
        )
        await update.message.reply_text(error_message, parse_mode="Markdown")

async def handle_schema_confirmation(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: ConversationSession
) -> None:
    """
    Handle schema confirmation and proceed to target selection.

    User Input: "yes" or "looks good" → SELECTING_TARGET
    User Input: "retry" or "no" → AWAITING_FILE_PATH
    """
    user_input = update.message.text.strip().lower()

    if user_input in ["yes", "y", "looks good", "good", "ok", "continue"]:
        # Load data from path
        file_path = Path(session.selections["file_path"])

        try:
            df = load_local_file(file_path)
            session.data = df
            session.workflow_state = MLTrainingState.SELECTING_TARGET

            # Show target selection prompt
            target_prompt = build_target_selection_prompt(df.columns.tolist())
            await update.message.reply_text(target_prompt, parse_mode="Markdown")

        except Exception as e:
            error_message = (
                f"❌ **Failed to load dataset**\n\n"
                f"Error: {str(e)}\n\n"
                f"Type **retry** to provide a new path."
            )
            await update.message.reply_text(error_message, parse_mode="Markdown")

    elif user_input in ["retry", "no", "n", "wrong", "change"]:
        # Go back to file path input
        session.workflow_state = MLTrainingState.AWAITING_FILE_PATH

        config = context.bot_data.get("config")
        allowed_dirs = config.get("local_data", {}).get("allowed_directories", [])

        message = (
            "🔄 **Retry File Path**\n\n" +
            FILE_PATH_PROMPT.format(
                allowed_dirs="\n".join(f"• `{d}`" for d in allowed_dirs)
            )
        )
        await update.message.reply_text(message, parse_mode="Markdown")

    else:
        # Invalid input
        message = (
            "⚠️ Invalid response.\n\n"
            "Reply **'yes'** to continue or **'retry'** to provide new path."
        )
        await update.message.reply_text(message, parse_mode="Markdown")

# Update state router
async def route_workflow_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    session: ConversationSession
) -> None:
    """Route messages based on current workflow state."""
    state = session.workflow_state

    handlers = {
        MLTrainingState.AWAITING_DATA_SOURCE: handle_data_source_selection,  # NEW
        MLTrainingState.AWAITING_FILE_PATH: handle_file_path_input,  # NEW
        MLTrainingState.CONFIRMING_SCHEMA: handle_schema_confirmation,  # NEW
        MLTrainingState.AWAITING_DATA: handle_telegram_upload,  # Existing
        MLTrainingState.SELECTING_TARGET: handle_target_selection,  # Existing
        MLTrainingState.SELECTING_FEATURES: handle_feature_selection,  # Existing
        MLTrainingState.CONFIRMING_MODEL: handle_model_confirmation,  # Existing
        MLTrainingState.SPECIFYING_ARCHITECTURE: handle_architecture_specification,  # Existing
        MLTrainingState.COLLECTING_HYPERPARAMETERS: handle_hyperparameter_collection,  # Existing
        # ... rest of existing handlers
    }

    handler = handlers.get(state)
    if handler:
        await handler(update, context, session)
    else:
        logger.warning(f"No handler for state: {state}")
```

**Modify Train Command** (`src/bot/handlers.py`):

```python
async def train_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE
) -> None:
    """
    Enhanced /train command with data source selection.

    NEW Flow (if local_data enabled):
    1. Check config for local_data.enabled
    2. If enabled → Show data source options
    3. If disabled → Use existing Telegram upload flow
    """
    user_id = update.effective_user.id

    # Get or create session
    session = get_or_create_session(user_id, context)

    # Check if local data feature enabled
    config = context.bot_data.get("config", {})
    local_data_enabled = config.get("local_data", {}).get("enabled", False)

    if local_data_enabled:
        # NEW: Show data source selection
        message = DATA_SOURCE_PROMPT
        session.workflow_state = MLTrainingState.AWAITING_DATA_SOURCE
    else:
        # Existing: Direct to Telegram upload
        message = (
            "📤 **Upload Dataset**\n\n"
            "Please upload your dataset file.\n\n"
            "**Supported formats:** CSV, Excel, Parquet\n"
            "**Max size:** 50 MB"
        )
        session.workflow_state = MLTrainingState.AWAITING_DATA

    await update.message.reply_text(message, parse_mode="Markdown")
```

#### Test Cases (`tests/integration/test_local_path_workflow.py`)
```python
@pytest.mark.asyncio
class TestLocalPathWorkflow:
    """Integration tests for local path training workflow."""

    async def test_complete_local_path_workflow(self):
        """Complete workflow: source selection → path → schema → training."""

    async def test_data_source_selection_telegram(self):
        """Selecting Telegram should route to existing upload flow."""

    async def test_data_source_selection_local_path(self):
        """Selecting local path should prompt for file path."""

    async def test_valid_path_shows_schema(self):
        """Valid path should detect and display schema."""

    async def test_invalid_path_shows_error(self):
        """Invalid path should show helpful error message."""

    async def test_schema_confirmation_yes(self):
        """Confirming schema should load data and proceed to target selection."""

    async def test_schema_confirmation_retry(self):
        """Retry should return to file path input."""

    async def test_path_traversal_blocked(self):
        """Path traversal attempts should be blocked with error."""

    async def test_oversized_file_rejected(self):
        """Files exceeding size limit should be rejected."""

    async def test_telegram_workflow_unchanged(self):
        """Telegram upload workflow should still work (backward compatibility)."""

    async def test_large_file_performance(self):
        """Large files (>100MB) should load efficiently."""

    async def test_cancel_command_works(self):
        """/cancel should work at any workflow state."""
```

#### Acceptance Criteria
- ✅ Data source selection works (Telegram vs Local)
- ✅ File path validation provides clear errors
- ✅ Schema detection displays correctly
- ✅ Schema confirmation routes correctly
- ✅ Backward compatibility maintained (Telegram uploads work)
- ✅ All integration tests passing

---

### **Phase 6: UI/UX Polish** 🟡 MEDIUM
**Duration**: 1 hour | **Priority**: MEDIUM

#### Prompt Templates

```python
DATA_SOURCE_PROMPT = """
🚀 **ML Training Workflow**

📂 **Choose Data Source:**

1️⃣ **Upload via Telegram** (recommended for files <50MB)
   • Easy and quick
   • No configuration needed

2️⃣ **Local File Path** (for large datasets >50MB)
   • Faster for big files
   • Keep data private (not sent to Telegram)
   • Requires file to be on server

Reply with: **1** or **2**

Type **/cancel** to cancel workflow.
"""

FILE_PATH_PROMPT = """
📁 **Provide Local File Path**

Enter the full path to your dataset:

**Examples:**
• macOS: `/Users/username/datasets/data.csv`
• Linux: `/home/username/data/data.xlsx`
• Windows: `C:\\Users\\username\\data.csv`

**Allowed directories:**
{allowed_dirs}

**Supported formats:** CSV, Excel (.xlsx, .xls), Parquet

Type **/cancel** to cancel workflow.
"""
```

#### Error Messages

```python
def format_path_error(
    error_msg: str,
    allowed_dirs: list[str],
    max_size_mb: int,
    allowed_exts: list[str]
) -> str:
    """Format path validation error with helpful guidance."""

    if "not in allowed directories" in error_msg.lower():
        return (
            "⚠️ **Path Not Allowed**\n\n"
            "The file path is outside allowed directories.\n\n"
            "**Allowed directories:**\n" +
            "\n".join(f"• `{d}`" for d in allowed_dirs) +
            "\n\nPlease provide a path within these directories."
        )

    elif "file not found" in error_msg.lower():
        return (
            "❌ **File Not Found**\n\n"
            f"Could not find file.\n\n"
            "Please check:\n"
            "• Path is correct (no typos)\n"
            "• File exists at this location\n"
            "• You have permission to read it"
        )

    elif "file too large" in error_msg.lower():
        return (
            "⚠️ **File Too Large**\n\n"
            f"Maximum file size: {max_size_mb} MB\n\n"
            "Please use a smaller file or contact admin."
        )

    elif "invalid extension" in error_msg.lower():
        return (
            "❌ **Invalid File Type**\n\n"
            f"Supported formats: {', '.join(allowed_exts)}\n\n"
            "Please provide a CSV, Excel, or Parquet file."
        )

    else:
        return f"❌ **Validation Error**\n\n{error_msg}"
```

---

### **Phase 7: Testing & Validation** 🔴 CRITICAL
**Duration**: 2-3 hours | **Priority**: CRITICAL

#### Test Coverage Requirements
- **Security tests**: 100% coverage (path validation CRITICAL)
- **Unit tests**: 90%+ coverage (all new modules)
- **Integration tests**: Complete workflows (Telegram + Local path)

#### Test Datasets (Create `tests/fixtures/test_datasets/`)
```bash
# Small dataset (for quick tests)
tests/fixtures/test_datasets/small_dataset.csv  # 1KB, 100 rows

# Medium dataset (for performance tests)
tests/fixtures/test_datasets/medium_dataset.csv  # 10MB, 100k rows

# Large dataset (for chunking tests)
tests/fixtures/test_datasets/large_dataset.csv  # 100MB, 1M rows

# Edge cases
tests/fixtures/test_datasets/corrupted_file.csv  # Malformed data
tests/fixtures/test_datasets/empty_file.csv  # Empty file
tests/fixtures/test_datasets/wrong_extension.txt  # CSV with wrong extension
tests/fixtures/test_datasets/unicode_columns.csv  # Unicode column names
tests/fixtures/test_datasets/missing_values.csv  # Heavy missing data
```

#### Security Test Suite (`tests/unit/test_path_security.py`)
```python
class TestPathSecurity:
    """CRITICAL: Security validation tests."""

    def test_path_traversal_parent_blocked(self):
        """../../etc/passwd should be blocked."""

    def test_path_traversal_encoded_blocked(self):
        """%2e%2e%2fetc%2fpasswd should be blocked."""

    def test_symlink_outside_whitelist_blocked(self):
        """Symlinks pointing outside whitelist should be blocked."""

    def test_absolute_path_only(self):
        """Relative paths should be normalized to absolute."""

    def test_whitelist_strictly_enforced(self):
        """Only whitelisted directories should be accessible."""

    def test_no_directory_traversal_bypass(self):
        """/allowed/subdir/../../../etc should be blocked."""

    def test_windows_path_traversal_blocked(self):
        """..\\..\\windows\\system32 should be blocked."""

    def test_hidden_files_validated_same(self):
        """Hidden files (.file) should follow same rules."""

    def test_special_characters_handled(self):
        """Paths with spaces, unicode, etc. should work."""

    def test_case_sensitivity_correct(self):
        """Case sensitivity should match OS (case-insensitive on macOS/Windows)."""

    def test_permission_denied_handled_gracefully(self):
        """Permission errors should not crash, just error message."""

    def test_zero_byte_file_rejected(self):
        """Empty files should be rejected."""

    def test_file_size_limit_enforced(self):
        """Files exceeding max size should be rejected."""

    def test_invalid_extension_rejected(self):
        """Non-CSV/Excel/Parquet should be rejected."""

    def test_disguised_file_type_detected(self):
        """.txt file with CSV content should be rejected (extension-based)."""
```

#### Integration Test Scenarios
```python
# Complete workflows
- ✅ Happy path: Local path → schema → target → features → train
- ✅ Happy path: Telegram upload → target → features → train
- ❌ Invalid path → error → retry → success
- ❌ Schema rejected → retry path → success
- ❌ Oversized file → error → provide different file
- ❌ Path traversal attempt → blocked → error shown
- ✅ Large file (>100MB) → chunked loading → success
- ✅ Cancel workflow at any step → clean session reset
```

#### Performance Benchmarks
```python
def test_small_file_performance():
    """Small files (<1MB) should load in <100ms."""

def test_medium_file_performance():
    """Medium files (10MB) should load in <1s."""

def test_large_file_performance():
    """Large files (100MB) should load in <10s with chunking."""

def test_schema_detection_performance():
    """Schema detection should complete in <500ms for typical files."""
```

---

### **Phase 8: Documentation** 🟡 MEDIUM
**Duration**: 1 hour | **Priority**: MEDIUM

#### Update CLAUDE.md

Add section: **"Local File Path Training Workflow"**

```markdown
### Local File Path Training Workflow

**Purpose**: Train ML models using local file paths for large datasets (>50MB) or sensitive data.

**When to Use**:
- Datasets larger than 50MB (Telegram limit)
- Sensitive data that shouldn't pass through Telegram
- Faster training with local file access
- Datasets already on server filesystem

**Security**:
- Strict directory whitelisting (configured in `config/config.yaml`)
- Multi-layer path validation (traversal prevention, size limits)
- File type validation (CSV, Excel, Parquet only)

**Configuration** (`config/config.yaml`):
```yaml
local_data:
  enabled: true  # Feature flag
  allowed_directories:
    - "/path/to/datasets"
    - "/another/path"
  max_file_size_mb: 1000  # 1GB
  allowed_extensions:
    - ".csv"
    - ".xlsx"
    - ".xls"
    - ".parquet"
```

**User Flow**:
1. `/train` → Choose "2. Local file path"
2. Provide path: `/Users/username/datasets/data.csv`
3. Review auto-detected schema (columns, types, statistics)
4. Confirm "yes" to proceed
5. Continue with normal workflow (target, features, model type)
6. Train model

**Developer Implementation**:
- **Path Validation**: `src/utils/path_validator.py` - Multi-layer security checks
- **Schema Detection**: `src/processors/schema_detector.py` - Auto type inference
- **Data Loading**: `src/processors/data_loader.py` - Chunked loading for large files
- **Workflow Handlers**: `src/bot/workflow_handlers.py` - State management
- **States**: `AWAITING_DATA_SOURCE`, `AWAITING_FILE_PATH`, `CONFIRMING_SCHEMA`

**Security Considerations**:
- NEVER disable path validation
- ALWAYS use strict whitelist (don't allow `/` or `~`)
- Test thoroughly with security test suite
- Review allowed_directories in production

**Troubleshooting**:
- "Path not allowed" → Check `allowed_directories` in config
- "File too large" → Increase `max_file_size_mb` or use smaller file
- "File not found" → Verify path is absolute and file exists
- Permission errors → Check file read permissions
```

#### Create User Guide

**File**: `docs/user_guide_local_paths.md`

```markdown
# User Guide: Local File Path Training

## Overview

This guide explains how to train ML models using local file paths instead of uploading files through Telegram.

## When to Use Local Paths

✅ **Use local paths when:**
- Your dataset is larger than 50MB (Telegram's file size limit)
- You want to keep data private (not sent through Telegram servers)
- Dataset is already on the server where the bot runs
- You need faster training with direct file access

❌ **Use Telegram upload when:**
- Dataset is small (<50MB)
- You don't have access to the server filesystem
- Quick one-off analysis

## Step-by-Step Guide

### 1. Start Training Workflow

Send `/train` command to the bot.

### 2. Choose Data Source

Bot will ask you to choose between:
- **1. Upload via Telegram** (traditional method)
- **2. Provide local file path** (for large files)

Reply: `2`

### 3. Provide File Path

Bot will show allowed directories and ask for file path.

**Example paths:**
- macOS: `/Users/yourusername/Documents/datasets/data.csv`
- Linux: `/home/yourusername/data/sales_data.xlsx`
- Windows: `C:\Users\yourusername\data\dataset.parquet`

**Important**:
- Must be absolute path (not relative)
- Must be within allowed directories
- File must exist and be readable

### 4. Review Schema

Bot will automatically detect and show:
- Number of rows and columns
- Column names and types (numeric, text, category)
- Missing value statistics
- Sample values from each column

**Example output:**
```
📊 Dataset Schema Detected

📁 File: german_credit_data.csv
📐 Shape: 799 rows × 21 columns
💾 Size: 63.3 KB

📋 Columns:
1. ✅ age (numeric) - 799 values, 0% missing
   Sample: [25, 30, 35]
2. ✅ income (numeric) - 799 values, 0% missing
   Sample: [30000, 45000, 55000]
...

✅ Ready for training? Reply 'yes' to continue.
```

### 5. Confirm or Retry

- If schema looks correct: Reply `yes`
- If wrong file: Reply `retry` to provide new path
- To cancel: Send `/cancel`

### 6. Continue Normal Workflow

After confirming schema:
1. Select target column
2. Select feature columns
3. Choose model type
4. Train model

## Troubleshooting

### "Path not in allowed directories"

**Problem**: The file path you provided is outside configured directories.

**Solution**:
- Check with bot admin which directories are allowed
- Move your dataset to an allowed directory
- Ask admin to add your directory to whitelist

### "File not found"

**Problem**: Bot cannot find the file at provided path.

**Solutions**:
- Verify path is correct (no typos)
- Ensure file exists at that location
- Use absolute path (not relative)
- Check file hasn't been moved or deleted

### "File too large"

**Problem**: File exceeds maximum size limit.

**Solutions**:
- Use a smaller subset of data
- Ask admin to increase size limit
- Use file compression or sampling

### "Invalid file type"

**Problem**: File extension is not supported.

**Supported formats**:
- CSV: `.csv`
- Excel: `.xlsx`, `.xls`
- Parquet: `.parquet`

**Solution**: Convert your data to one of these formats.

### Permission Errors

**Problem**: Bot cannot read the file.

**Solutions**:
- Check file permissions (should be readable)
- On Unix: `chmod 644 yourfile.csv`
- Ensure file is not locked by another program

## Security & Privacy

### What Gets Shared
- ✅ File path (sent to bot for validation)
- ✅ Schema information (column names, types)
- ❌ Actual data (NEVER sent through Telegram)

### Directory Whitelisting
The bot can ONLY access files in pre-configured directories. This prevents:
- Accessing system files
- Reading sensitive data from other locations
- Path traversal attacks

### Data Privacy
- Data never leaves the server
- Telegram servers never see your dataset
- Only you and the bot can see results

## Best Practices

1. **Organize datasets**: Keep datasets in dedicated allowed directories
2. **Use descriptive names**: `sales_2024_cleaned.csv` better than `data.csv`
3. **Check schema carefully**: Verify column types before training
4. **Clean data first**: Remove duplicates, handle missing values
5. **Test with small subset**: Use small file first to test workflow

## Example Session

```
You: /train

Bot: 🚀 ML Training Workflow
     Choose data source:
     1️⃣ Upload via Telegram
     2️⃣ Local file path

You: 2

Bot: 📁 Provide Local File Path
     Allowed directories:
     • /Users/yourname/datasets

You: /Users/yourname/datasets/housing.csv

Bot: 📊 Dataset Schema Detected
     799 rows × 8 columns
     [shows schema details]

     Ready for training? Reply 'yes'

You: yes

Bot: 🎯 Select Target Column:
     1. price
     2. sqft
     3. bedrooms
     [continues workflow...]
```

## Getting Help

- **Check schema**: Carefully review detected column types
- **Test small first**: Use small subset before full dataset
- **Ask admin**: Contact bot admin about directory access
- **Cancel anytime**: Use `/cancel` to stop workflow

## Administrator Configuration

If you're setting up the bot, see `CLAUDE.md` for:
- Configuration file setup
- Security best practices
- Directory whitelisting
- Size limit configuration
```

---

## 📊 Success Criteria

### Functional Requirements
- ✅ Users can provide local file paths for training
- ✅ Auto-schema detection works for 95%+ of datasets
- ✅ Workflow UX matches existing `/train` quality
- ✅ Backward compatibility: Telegram upload still works
- ✅ Supports CSV, Excel, Parquet files
- ✅ Handles files up to 1GB (configurable)

### Security Requirements
- ✅ Zero path traversal vulnerabilities
- ✅ Strict directory whitelist enforcement
- ✅ File size and type validation
- ✅ 100% test coverage for security validation
- ✅ No unauthorized file access possible

### Quality Requirements
- ✅ 90%+ unit test coverage
- ✅ All integration tests passing
- ✅ No performance regression (<100ms overhead for small files)
- ✅ Comprehensive error handling
- ✅ Clear user feedback at every step

---

## ⚠️ Risk Assessment & Mitigation

### 🔴 High Risk: Security Vulnerabilities

**Risk**: Path traversal attacks, unauthorized file access, malicious file exploitation

**Impact**: CRITICAL - Could expose sensitive system files or user data

**Mitigation**:
1. Multi-layer validation (path, file, content)
2. Strict directory whitelist (no exceptions)
3. Extensive security testing (15+ attack scenarios)
4. Feature flag for easy emergency disable
5. Security audit before production deployment

**Validation**: 100% security test pass rate required

---

### 🟡 Medium Risk: User Experience Complexity

**Risk**: Users confused by new workflow, unclear error messages, too many steps

**Impact**: MEDIUM - Could lead to support burden, user frustration

**Mitigation**:
1. Auto-schema detection (minimize user input)
2. Clear prompts with examples
3. Helpful error messages with solutions
4. Backward compatible (existing flow preserved)
5. User guide documentation

**Validation**: User testing with 3+ scenarios

---

### 🟡 Medium Risk: Configuration Errors

**Risk**: Incorrect `allowed_directories`, missing config, wrong file paths

**Impact**: MEDIUM - Feature won't work, users blocked

**Mitigation**:
1. Validation on config load
2. Clear documentation (CLAUDE.md + user guide)
3. Sensible defaults (empty allowed_directories = disabled)
4. Startup checks (log allowed directories)
5. Error messages guide fixes

**Validation**: Config validation tests pass

---

### 🟢 Low Risk: Performance Degradation

**Risk**: Large file loading slows down system, memory issues

**Impact**: LOW - Slower responses, but not blocking

**Mitigation**:
1. Chunked loading for files >100MB
2. Configurable memory limits
3. Sampling for schema detection (not full load)
4. Performance benchmarks in tests

**Validation**: Performance tests pass (<10s for 100MB files)

---

## 📈 Estimated Timeline

| Phase | Tasks | Duration | Priority | Dependencies |
|-------|-------|----------|----------|--------------|
| **Phase 1** | Security foundation (path validator + tests) | 1-2 hours | 🔴 CRITICAL | None |
| **Phase 2** | Schema detection (auto-inference + formatting) | 1-2 hours | 🔴 HIGH | Phase 1 |
| **Phase 3** | Data loader enhancement (local path support) | 1 hour | 🔴 HIGH | Phase 1 |
| **Phase 4** | State management (4 new states + transitions) | 30 min | 🟡 MEDIUM | None |
| **Phase 5** | Workflow handlers (3 new handlers + integration) | 2-3 hours | 🔴 HIGH | Phases 1-4 |
| **Phase 6** | UI/UX polish (prompts + error messages) | 1 hour | 🟡 MEDIUM | Phase 5 |
| **Phase 7** | Testing (unit + integration + security) | 2-3 hours | 🔴 CRITICAL | All phases |
| **Phase 8** | Documentation (CLAUDE.md + user guide) | 1 hour | 🟡 MEDIUM | All phases |
| **TOTAL** | **8 phases** | **10-14 hours** | | |

### Critical Path
**Phase 1 → Phase 2 → Phase 3 → Phase 5 → Phase 7**

These phases must be completed sequentially and cannot be parallelized.

### Parallel Work Opportunities
- Phase 4 (state management) can be done in parallel with Phases 2-3
- Phase 6 (UI/UX) can start once Phase 5 handlers are sketched out
- Phase 8 (docs) can be written while Phase 7 tests run

---

## 🔄 Alternative Approaches (Considered & Rejected)

### ❌ Approach 1: Separate `/algorithm` Command

**Rejected Reasoning**:
- Code duplication (two entry points for similar workflows)
- User confusion (when to use `/train` vs `/algorithm`?)
- Maintenance burden (two workflows to update)

**Decision**: Enhance existing `/train` command instead

---

### ❌ Approach 2: Manual JSON Schema Entry

**Rejected Reasoning**:
- Poor UX (requires technical knowledge of JSON)
- Error-prone (typos in JSON common)
- Time-consuming for users
- High support burden

**Decision**: Auto-detection with visual confirmation instead

---

### ❌ Approach 3: No Schema Confirmation Step

**Rejected Reasoning**:
- Risk of incorrect type inference (e.g., category detected as text)
- Leads to training errors downstream
- Users can't verify before committing to long training

**Decision**: Always show detected schema for user verification

---

### ❌ Approach 4: Direct File System Access (No Whitelist)

**Rejected Reasoning**:
- CRITICAL security risk (path traversal, system file access)
- Unacceptable in production environment
- Violates principle of least privilege

**Decision**: Strict directory whitelisting with multi-layer validation

---

## 📝 Implementation Notes

### Key Design Decisions

1. **Security First**: Strict validation before any file access
   - Rationale: Security vulnerabilities would be CRITICAL severity
   - Trade-off: Slightly more complex setup (whitelist configuration)

2. **UX Optimized**: Auto-detection minimizes user input
   - Rationale: Compete with ease of Telegram uploads
   - Trade-off: Potential type inference errors (mitigated by confirmation)

3. **Backward Compatible**: Existing Telegram flow unchanged
   - Rationale: Don't break existing user workflows
   - Trade-off: Slightly more complex routing logic

4. **Feature Flagged**: Can be disabled via configuration
   - Rationale: Easy rollback if issues in production
   - Trade-off: Need to test both enabled/disabled states

5. **Fail-Safe**: Comprehensive error handling at every step
   - Rationale: Users should never see stack traces
   - Trade-off: More code for error formatting

### Dependencies

**Required**:
- `pandas` - Already in requirements.txt
- `pathlib` - Standard library
- `python-telegram-bot` - Already in requirements.txt

**Optional** (for file formats):
- `openpyxl` - Already in requirements.txt (Excel support)
- `pyarrow` - Already in requirements.txt (Parquet support)

**No new dependencies needed** ✅

### Configuration Management

**Development** (`config/config.yaml`):
```yaml
local_data:
  enabled: true
  allowed_directories:
    - "./data"
    - "./tests/fixtures/test_datasets"
  max_file_size_mb: 100
```

**Production** (`config/config.yaml`):
```yaml
local_data:
  enabled: true
  allowed_directories:
    - "/var/data/user_datasets"
  max_file_size_mb: 1000
```

**Testing** (`config/config.test.yaml`):
```yaml
local_data:
  enabled: true
  allowed_directories:
    - "./tests/fixtures/test_datasets"
  max_file_size_mb: 10  # Small for fast tests
```

---

## ✅ Validation Checklist

Before marking this feature complete, validate:

### Functional Validation
- [ ] Users can select "Local file path" option
- [ ] Valid paths show schema detection
- [ ] Invalid paths show clear errors
- [ ] Schema displays columns, types, statistics
- [ ] Confirming schema loads data and proceeds
- [ ] Retry returns to file path input
- [ ] Training completes with local path data
- [ ] Telegram upload flow still works (backward compatibility)

### Security Validation
- [ ] Path traversal attempts blocked (../../etc/passwd)
- [ ] Paths outside whitelist rejected
- [ ] Symlinks outside whitelist blocked
- [ ] Oversized files rejected
- [ ] Invalid extensions rejected
- [ ] Relative paths normalized
- [ ] Special characters handled safely
- [ ] Permission errors handled gracefully
- [ ] All 15+ security tests passing

### Quality Validation
- [ ] 90%+ unit test coverage
- [ ] All integration tests passing
- [ ] Performance benchmarks met:
  - Small files (<1MB): <100ms
  - Medium files (10MB): <1s
  - Large files (100MB): <10s
- [ ] No memory leaks with large files
- [ ] Error messages are clear and actionable

### Documentation Validation
- [ ] CLAUDE.md updated with new workflow
- [ ] User guide created (docs/user_guide_local_paths.md)
- [ ] Configuration documented
- [ ] Troubleshooting section complete
- [ ] Security best practices documented

---

## 🚀 Deployment Plan

### Pre-Deployment
1. Complete all 8 phases
2. Run full test suite (100% security tests pass)
3. Performance benchmarks pass
4. Code review (security focus)
5. Update documentation

### Deployment Steps
1. **Staging Environment**:
   - Deploy with `local_data.enabled: false`
   - Test Telegram upload flow (backward compatibility)
   - Enable local_data feature
   - Test complete workflow with test datasets
   - Monitor logs for errors

2. **Production Environment**:
   - Deploy with feature flag OFF initially
   - Configure `allowed_directories` (production paths)
   - Enable feature flag
   - Monitor first 10 user sessions closely
   - Watch for security events

### Monitoring
- **Security Alerts**: Path traversal attempts, whitelist violations
- **Performance Metrics**: File load times, memory usage
- **Error Rates**: Path validation failures, schema detection errors
- **User Adoption**: % users choosing local path vs Telegram

### Rollback Plan
If critical issues found:
1. Set `local_data.enabled: false` in config
2. Restart bot (feature disabled)
3. Telegram upload flow continues working
4. Investigate and fix issues
5. Re-deploy when ready

---

## 📚 References

### Related Files
- `docs/prd.md` - Original product requirements
- `CLAUDE.md` - Project coding standards
- `dev/implemented/README.md` - Implementation history
- `src/core/state_manager.py` - State management system
- `src/bot/workflow_handlers.py` - Existing workflow handlers
- `src/processors/data_loader.py` - Data loading system

### Related Issues/PRs
- (To be created) Issue #X: Local file path training feature
- (To be created) PR #Y: Implement local path workflow

---

**Status**: ✅ Planning Complete, Ready for Implementation
**Next Step**: Begin Phase 1 (Security Foundation)
**Created**: 2025-10-06
**Last Updated**: 2025-10-06
