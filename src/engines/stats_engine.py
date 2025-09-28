"""
Statistical Analysis Engine for the Statistical Modeling Agent.

This module provides comprehensive statistical analysis capabilities including
descriptive statistics and correlation analysis with robust missing data handling.
"""

import logging
import warnings
from typing import Any, Dict, List, Optional, Literal, Union
import pandas as pd
import numpy as np
from scipy import stats

from src.core.parser import TaskDefinition
from src.utils.exceptions import DataError, ValidationError
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Type aliases for better readability
MissingStrategy = Literal["drop", "mean", "median", "zero", "forward"]
CorrelationMethod = Literal["pearson", "spearman", "kendall"]
StatisticsList = List[Literal["mean", "median", "std", "min", "max", "count", "quartiles"]]


class StatsEngine:
    """Statistical analysis engine for descriptive statistics and correlations."""

    # Statistics function mapping for efficient calculation
    STAT_FUNCTIONS = {
        "mean": lambda data: data.mean(),
        "median": lambda data: data.median(),
        "std": lambda data: data.std(),
        "min": lambda data: data.min(),
        "max": lambda data: data.max(),
        "count": lambda data: int(data.count())
    }

    # Missing data strategy mapping
    MISSING_STRATEGIES = {
        "drop": lambda data: data.dropna(),
        "mean": lambda data: data.fillna(data.mean()),
        "median": lambda data: data.fillna(data.median()),
        "zero": lambda data: data.fillna(0),
        "forward": lambda data: data.fillna(method='ffill')
    }

    def __init__(self, default_precision: int = 4) -> None:
        """Initialize the statistics engine."""
        self.logger = logger
        self.default_precision = default_precision
        self.logger.info("StatsEngine initialized")

    def execute(self, task: TaskDefinition, data: pd.DataFrame) -> Dict[str, Any]:
        """Main entry point for orchestrator integration."""
        self.logger.info(f"Executing stats task: {task.operation}")

        # Validate inputs
        self._validate_task(task)
        self._validate_dataframe(data)

        # Route to appropriate method based on operation
        operation = task.operation.lower()
        parameters = task.parameters or {}

        try:
            if operation in ["descriptive_stats", "summary_analysis", "mean_analysis", "median_analysis", "std_analysis"]:
                return self.calculate_descriptive_stats(data, **parameters)
            elif operation in ["correlation_analysis", "correlation"]:
                return self.calculate_correlation(data, **parameters)
            else:
                raise ValidationError(
                    f"Unsupported operation: {operation}",
                    field="operation",
                    value=operation
                )

        except Exception as e:
            self.logger.error(f"Error executing {operation}: {str(e)}")
            raise

    def calculate_descriptive_stats(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        statistics: Optional[StatisticsList] = None,
        missing_strategy: MissingStrategy = "mean",
        precision: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics.

        Args:
            data: pandas DataFrame to analyze
            columns: List of column names or None for all numeric columns
            statistics: List of statistics to compute or None for all
            missing_strategy: Strategy for handling missing data
            precision: Number of decimal places (uses default if None)

        Returns:
            Dictionary with descriptive statistics and metadata
        """
        self.logger.info("Calculating descriptive statistics")
        precision = precision or self.default_precision

        # Prepare columns
        target_columns = self._prepare_columns(data, columns, require_numeric=True)

        # Handle missing data
        processed_data = self._handle_missing_data(data[target_columns], missing_strategy)

        # Calculate statistics
        if statistics is None or (statistics and "summary" in statistics):
            statistics = ["mean", "median", "std", "min", "max", "count", "quartiles"]

        results = {}
        for column in processed_data.columns:
            column_data = processed_data[column]
            column_results = {}

            for stat in statistics:
                try:
                    if stat in self.STAT_FUNCTIONS:
                        result = self.STAT_FUNCTIONS[stat](column_data)
                        column_results[stat] = round(result, precision) if stat != "count" else result
                    elif stat == "quartiles":
                        q1, q3 = column_data.quantile([0.25, 0.75])
                        column_results["quartiles"] = {"q1": round(q1, precision), "q3": round(q3, precision)}
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {stat} for {column}: {e}")
                    column_results[stat] = None

            # Add missing data info
            original_count = len(data[column])
            processed_count = len(column_data)
            column_results["missing"] = original_count - processed_count

            results[column] = column_results

        # Generate metadata
        metadata = {
            "summary": {
                "total_columns": len(target_columns),
                "numeric_columns": len(target_columns),
                "missing_strategy": missing_strategy,
                "statistics_computed": statistics,
                "precision": precision
            }
        }

        results.update(metadata)
        self.logger.info(f"Descriptive statistics calculated for {len(target_columns)} columns")
        return results

    def calculate_correlation(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]] = None,
        method: CorrelationMethod = "pearson",
        min_periods: int = 1,
        significance_threshold: float = 0.5,
        precision: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calculate correlation matrix and analysis.

        Args:
            data: pandas DataFrame to analyze
            columns: List of column names or None for all numeric columns
            method: Correlation method (pearson, spearman, kendall)
            min_periods: Minimum observations required for correlation
            significance_threshold: Threshold for significant correlations
            precision: Number of decimal places (uses default if None)

        Returns:
            Dictionary with correlation matrix and analysis results
        """
        self.logger.info(f"Calculating {method} correlation matrix")
        precision = precision or self.default_precision

        # Prepare columns
        target_columns = self._prepare_columns(data, columns, require_numeric=True)

        if len(target_columns) < 2:
            raise DataError(
                "Correlation analysis requires at least 2 numeric columns",
                data_shape=data.shape
            )

        # Handle missing data (use drop strategy for correlations)
        processed_data = self._handle_missing_data(data[target_columns], "drop")

        if len(processed_data) < min_periods:
            raise DataError(
                f"Insufficient data for correlation: {len(processed_data)} rows, minimum {min_periods} required",
                data_shape=processed_data.shape
            )

        # Calculate correlation matrix
        try:
            if method == "pearson":
                corr_matrix = processed_data.corr(method='pearson', min_periods=min_periods)
            elif method == "spearman":
                corr_matrix = processed_data.corr(method='spearman', min_periods=min_periods)
            elif method == "kendall":
                corr_matrix = processed_data.corr(method='kendall', min_periods=min_periods)
            else:
                raise ValidationError(
                    f"Unsupported correlation method: {method}",
                    field="method",
                    value=method
                )
        except Exception as e:
            raise DataError(f"Failed to calculate correlation matrix: {str(e)}")

        # Round correlation values
        corr_matrix = corr_matrix.round(precision)

        # Find significant correlations
        significant_correlations = []
        strongest_correlation = {"pair": None, "value": 0.0}

        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates and self-correlation
                    corr_value = corr_matrix.loc[col1, col2]

                    if not pd.isna(corr_value):
                        abs_corr = abs(corr_value)

                        # Track strongest correlation
                        if abs_corr > abs(strongest_correlation["value"]):
                            strongest_correlation = {
                                "pair": (col1, col2),
                                "value": corr_value
                            }

                        # Track significant correlations
                        if abs_corr >= significance_threshold:
                            significant_correlations.append({
                                "column1": col1,
                                "column2": col2,
                                "correlation": corr_value
                            })

        # Sort significant correlations by absolute value
        significant_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        # Build results
        results = {
            "correlation_matrix": corr_matrix.to_dict(),
            "significant_correlations": significant_correlations,
            "summary": {
                "method": method,
                "total_pairs": len(target_columns) * (len(target_columns) - 1) // 2,
                "significant_pairs": len(significant_correlations),
                "strongest_correlation": strongest_correlation,
                "min_periods": min_periods,
                "significance_threshold": significance_threshold,
                "precision": precision
            }
        }

        self.logger.info(f"Correlation analysis complete: {len(significant_correlations)} significant pairs found")
        return results

    def _handle_missing_data(
        self,
        data: pd.DataFrame,
        strategy: MissingStrategy
    ) -> pd.DataFrame:
        """
        Apply missing data handling strategy.

        Args:
            data: DataFrame with potential missing values
            strategy: Strategy to use for missing data

        Returns:
            DataFrame with missing data handled
        """
        if strategy in self.MISSING_STRATEGIES:
            return self.MISSING_STRATEGIES[strategy](data)
        else:
            raise ValidationError(
                f"Unsupported missing data strategy: {strategy}",
                field="missing_strategy",
                value=strategy
            )

    def _prepare_columns(
        self,
        data: pd.DataFrame,
        columns: Optional[List[str]],
        require_numeric: bool = False
    ) -> List[str]:
        """
        Prepare and validate column list.

        Args:
            data: DataFrame to analyze
            columns: Specified columns or None
            require_numeric: Whether to filter to numeric columns only

        Returns:
            List of validated column names
        """
        if columns is None or (len(columns) == 1 and columns[0] == "all"):
            target_columns = list(data.columns)
        else:
            # Validate specified columns exist
            missing_columns = [col for col in columns if col not in data.columns]
            if missing_columns:
                raise ValidationError(
                    f"Columns not found in data: {missing_columns}",
                    field="columns",
                    value=str(missing_columns)
                )
            target_columns = columns

        if require_numeric:
            # Filter to numeric columns only
            numeric_columns = []
            for col in target_columns:
                if pd.api.types.is_numeric_dtype(data[col]):
                    numeric_columns.append(col)
                else:
                    self.logger.warning(f"Skipping non-numeric column: {col}")

            if not numeric_columns:
                raise DataError(
                    "No numeric columns found for analysis",
                    data_shape=data.shape,
                    missing_columns=target_columns
                )

            target_columns = numeric_columns

        return target_columns

    def _validate_task(self, task: TaskDefinition) -> None:
        """Validate TaskDefinition object."""
        if not isinstance(task, TaskDefinition):
            raise ValidationError(
                "Invalid task type, expected TaskDefinition",
                field="task",
                value=str(type(task))
            )

        if task.task_type != "stats":
            raise ValidationError(
                f"Invalid task type for StatsEngine: {task.task_type}",
                field="task_type",
                value=task.task_type
            )

    def _validate_dataframe(self, data: pd.DataFrame) -> None:
        """Validate input DataFrame."""
        if not isinstance(data, pd.DataFrame):
            raise ValidationError(
                "Data must be a pandas DataFrame",
                field="data",
                value=str(type(data))
            )

        if data.empty:
            raise DataError(
                "DataFrame is empty",
                data_shape=(0, 0)
            )

        if len(data) == 0:
            raise DataError(
                "DataFrame has no rows",
                data_shape=data.shape
            )

        # Check for extreme missing data
        total_cells = data.shape[0] * data.shape[1]
        missing_cells = data.isnull().sum().sum()
        missing_percentage = (missing_cells / total_cells) * 100 if total_cells > 0 else 100

        if missing_percentage > 90:
            raise DataError(
                f"Too much missing data: {missing_percentage:.1f}% missing",
                data_shape=data.shape
            )
        elif missing_percentage > 50:
            self.logger.warning(f"High missing data: {missing_percentage:.1f}% missing")

        # Handle infinite values
        inf_cols = []
        for col in data.select_dtypes(include=[np.number]).columns:
            if np.isinf(data[col]).any():
                inf_cols.append(col)

        if inf_cols:
            self.logger.warning(f"Infinite values detected in columns: {inf_cols}")


# Convenience functions for backward compatibility
def calculate_mean(data: pd.DataFrame, column: str) -> float:
    """Calculate mean for a specific column."""
    engine = StatsEngine()
    result = engine.calculate_descriptive_stats(
        data, columns=[column], statistics=["mean"]
    )
    return result[column]["mean"]


def calculate_correlation_matrix(data: pd.DataFrame, method: str = "pearson") -> Dict[str, Any]:
    """Calculate correlation matrix for all numeric columns."""
    engine = StatsEngine()
    return engine.calculate_correlation(data, method=method)