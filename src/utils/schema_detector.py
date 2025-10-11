"""Automatic schema detection for datasets."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional
import pandas as pd


@dataclass
class ColumnSchema:
    """Schema information for a single column."""
    name: str
    dtype: str  # "numeric", "categorical", "datetime", "text"
    pandas_dtype: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: list[str]
    is_target_candidate: bool
    is_feature_candidate: bool


@dataclass
class DatasetSchema:
    """Complete dataset schema information."""
    file_path: str
    n_rows: int
    n_columns: int
    columns: list[ColumnSchema]
    suggested_task_type: Optional[Literal["regression", "classification"]]
    suggested_target: Optional[str]
    suggested_features: list[str]
    memory_usage_mb: float
    has_missing_values: bool
    overall_quality_score: float


def detect_schema(
    file_path: Path,
    max_sample_values: int = 5,
    auto_suggest: bool = True
) -> DatasetSchema:
    """Automatically detect dataset schema from file."""
    df = _load_dataframe(file_path)

    columns = [_analyze_column(df, col, max_sample_values) for col in df.columns]

    memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    has_missing = bool(df.isnull().any().any())

    # Auto-suggestions
    suggested_task = None
    suggested_target = None
    suggested_features = []

    if auto_suggest:
        suggested_task = _suggest_task_type(columns)
        suggested_target = _suggest_target(columns, suggested_task)
        suggested_features = _suggest_features(columns, suggested_target)

    # Simple quality score: completeness + size
    quality = (1.0 - sum(c.null_percentage for c in columns) / len(columns) / 100)
    quality *= 1.0 if len(df) >= 100 else 0.8
    quality *= 0.8 if len(columns) / len(df) > 0.1 else 1.0

    return DatasetSchema(
        file_path=str(file_path),
        n_rows=len(df),
        n_columns=len(df.columns),
        columns=columns,
        suggested_task_type=suggested_task,
        suggested_target=suggested_target,
        suggested_features=suggested_features,
        memory_usage_mb=memory_mb,
        has_missing_values=has_missing,
        overall_quality_score=min(1.0, quality)
    )


def _load_dataframe(file_path: Path) -> pd.DataFrame:
    """Load dataset from file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(file_path)
        elif suffix in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif suffix == ".parquet":
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {suffix}")
    except Exception as e:
        raise ValueError(f"Failed to load: {str(e)}") from e


def _analyze_column(df: pd.DataFrame, col_name: str, max_sample: int) -> ColumnSchema:
    """Analyze single column."""
    col = df[col_name]

    null_count = int(col.isnull().sum())
    null_pct = null_count / len(df) * 100
    unique_count = int(col.nunique())
    unique_pct = unique_count / len(df) * 100

    dtype = _infer_type(col)
    sample_values = [str(v) for v in col.dropna().unique()[:max_sample]]

    is_target = _is_candidate(col, dtype, unique_count, null_pct, "target")
    is_feature = _is_candidate(col, dtype, unique_count, null_pct, "feature")

    return ColumnSchema(
        name=col_name,
        dtype=dtype,
        pandas_dtype=str(col.dtype),
        null_count=null_count,
        null_percentage=null_pct,
        unique_count=unique_count,
        unique_percentage=unique_pct,
        sample_values=sample_values,
        is_target_candidate=is_target,
        is_feature_candidate=is_feature
    )


def _infer_type(col: pd.Series) -> str:
    """Infer high-level column type."""
    if pd.api.types.is_datetime64_any_dtype(col):
        return "datetime"

    if pd.api.types.is_numeric_dtype(col):
        return "numeric"

    if pd.api.types.is_object_dtype(col) or pd.api.types.is_categorical_dtype(col):
        unique_ratio = col.nunique() / len(col)
        return "categorical" if unique_ratio < 0.5 else "text"

    return "text"


def _is_candidate(col: pd.Series, dtype: str, unique_count: int, null_pct: float, role: str) -> bool:
    """Check if column is suitable for target or feature role."""
    # Reject high null percentage
    max_null = 5.0 if role == "target" else 80.0
    if null_pct > max_null:
        return False

    # Reject ID columns (sequential integers)
    non_null_count = len(col.dropna())
    if dtype == "numeric" and unique_count == non_null_count and non_null_count > 10:
        try:
            sorted_vals = sorted(col.dropna().unique())
            diffs = [sorted_vals[i+1] - sorted_vals[i] for i in range(len(sorted_vals)-1)]
            if all(d == 1 for d in diffs):
                return False
        except (TypeError, ValueError):
            pass

    # Target-specific rules
    if role == "target":
        if dtype == "numeric":
            return True
        if dtype == "categorical" and 2 <= unique_count <= 20:
            return True
        return False

    # Feature-specific rules
    if role == "feature":
        if dtype in ["numeric", "categorical", "datetime"]:
            return True
        return False

    return False


def _suggest_task_type(columns: list[ColumnSchema]) -> Optional[Literal["regression", "classification"]]:
    """Suggest ML task type."""
    targets = [c for c in columns if c.is_target_candidate]
    if not targets:
        return None

    cat_targets = [c for c in targets if c.dtype == "categorical" and 2 <= c.unique_count <= 10]
    if cat_targets:
        return "classification"

    num_targets = [c for c in targets if c.dtype == "numeric" and c.unique_percentage > 50.0]
    if num_targets:
        return "regression"

    return "regression" if any(c.dtype == "numeric" for c in targets) else "classification"


def _suggest_target(columns: list[ColumnSchema], task_type: Optional[str]) -> Optional[str]:
    """Suggest target column."""
    candidates = [c for c in columns if c.is_target_candidate]
    if not candidates:
        return None

    if task_type == "regression":
        candidates = [c for c in candidates if c.dtype == "numeric"]
    elif task_type == "classification":
        candidates = [c for c in candidates if c.dtype == "categorical"]

    if not candidates:
        return None

    target_keywords = ["target", "label", "class", "price", "value", "outcome", "y"]

    def score(col: ColumnSchema) -> float:
        s = 0.0
        if any(kw in col.name.lower() for kw in target_keywords):
            s += 10.0
        s += (100 - col.null_percentage) / 10
        if task_type == "classification" and 2 <= col.unique_count <= 10:
            s += 5.0
        return s

    return max(candidates, key=score).name


def _suggest_features(columns: list[ColumnSchema], target: Optional[str]) -> list[str]:
    """Suggest feature columns."""
    return [c.name for c in columns if c.is_feature_candidate and c.name != target]
