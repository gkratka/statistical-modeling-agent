# Stats Engine Implementation Plan

## Overview
The Statistics Engine (`src/engines/stats_engine.py`) is a core component that processes statistical analysis requests within the Statistical Modeling Agent. It integrates with the orchestrator to handle descriptive statistics and correlation analysis tasks.

## Architecture Integration

### System Flow Position
```
Parser → Orchestrator → StatsEngine → Results
```

The StatsEngine receives `TaskDefinition` objects from the orchestrator and returns formatted statistical results for downstream processing.

### Key Dependencies
- `src.core.parser.TaskDefinition` - Input task structure
- `src.utils.exceptions.DataError` - Data validation errors
- `src.utils.exceptions.ValidationError` - Input validation errors
- `src.utils.logger.get_logger` - Logging functionality
- `pandas.DataFrame` - Primary data structure

## Class Design

### StatsEngine Class
```python
class StatsEngine:
    """Statistical analysis engine for descriptive statistics and correlations."""

    def execute(self, task: TaskDefinition, data: pd.DataFrame) -> dict[str, Any]
    def calculate_descriptive_stats(self, data: pd.DataFrame, columns: list[str], **kwargs) -> dict[str, Any]
    def calculate_correlation(self, data: pd.DataFrame, columns: list[str], **kwargs) -> dict[str, Any]
```

### Core Methods

#### 1. execute(task: TaskDefinition, data: pd.DataFrame) → dict
**Purpose**: Main entry point for orchestrator integration
**Behavior**:
- Routes to appropriate statistical method based on `task.operation`
- Validates TaskDefinition parameters
- Handles error propagation with proper exception types
- Returns consistent result format

**Supported Operations**:
- `"descriptive_stats"` → `calculate_descriptive_stats()`
- `"correlation_analysis"` → `calculate_correlation()`
- `"mean_analysis"`, `"median_analysis"`, etc. → targeted descriptive stats

#### 2. calculate_descriptive_stats(data, columns, **kwargs) → dict
**Purpose**: Compute comprehensive descriptive statistics
**Parameters**:
- `data`: pandas DataFrame
- `columns`: list of column names or `["all"]`
- `missing_strategy`: `"drop"`, `"mean"`, `"median"`, `"zero"`, `"forward"`
- `statistics`: list of stats to compute (default: all)

**Returns**:
```python
{
    "column_name": {
        "mean": float,
        "median": float,
        "std": float,
        "min": float,
        "max": float,
        "count": int,
        "missing": int,
        "quartiles": {"q1": float, "q3": float}
    },
    "summary": {
        "total_columns": int,
        "numeric_columns": int,
        "missing_strategy": str
    }
}
```

#### 3. calculate_correlation(data, columns, **kwargs) → dict
**Purpose**: Generate correlation matrices and analysis
**Parameters**:
- `data`: pandas DataFrame
- `columns`: list of numeric column names or `["all"]`
- `method`: `"pearson"`, `"spearman"`, `"kendall"`
- `min_periods`: minimum observations for correlation

**Returns**:
```python
{
    "correlation_matrix": dict,  # column → column → correlation
    "significant_correlations": [
        {"column1": str, "column2": str, "correlation": float}
    ],
    "summary": {
        "method": str,
        "total_pairs": int,
        "significant_pairs": int,
        "strongest_correlation": {"pair": tuple, "value": float}
    }
}
```

## Missing Data Handling

### Strategies
1. **"drop"** (default for correlations): Remove rows with NaN values
2. **"mean"** (default for descriptive): Fill NaN with column mean
3. **"median"**: Fill NaN with column median
4. **"zero"**: Fill NaN with 0
5. **"forward"**: Forward fill NaN values

### Implementation
```python
def _handle_missing_data(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """Apply missing data strategy and return cleaned DataFrame."""
```

### Validation
- Track original vs. processed row counts
- Report missing data percentages
- Warn when >50% of data is missing
- Error when >90% of data is missing

## Error Handling

### Validation Checks
1. **DataFrame Validation**:
   - Non-empty DataFrame
   - At least one numeric column for correlations
   - Requested columns exist

2. **Column Validation**:
   - Column names are valid
   - Numeric columns for statistical operations
   - Handle mixed data types gracefully

3. **Edge Cases**:
   - Single-value columns (std = 0)
   - Constant columns (correlation undefined)
   - Infinite values (replace with NaN)

### Exception Hierarchy
```python
try:
    result = calculate_descriptive_stats(data, columns)
except ValidationError as e:
    # Input validation failed (missing columns, invalid parameters)
    pass
except DataError as e:
    # Data quality issues (too much missing data, invalid types)
    pass
```

## Output Formatting

### Numeric Precision
- Default: 4 decimal places
- Configurable via `precision` parameter
- Handle very large/small numbers appropriately

### Result Structure
All methods return dictionaries with:
- **Primary Results**: Core statistical computations
- **Metadata**: Method parameters and data quality info
- **Summary**: High-level insights and counts

## Integration Examples

### Orchestrator Integration
```python
# In orchestrator.py
if task.task_type == "stats":
    engine = StatsEngine()
    result = engine.execute(task, dataframe)
    return format_results(result)
```

### Parser Integration
```python
# TaskDefinition from parser
task = TaskDefinition(
    task_type="stats",
    operation="descriptive_stats",
    parameters={
        "columns": ["age", "income"],
        "statistics": ["mean", "median", "std"],
        "missing_strategy": "mean"
    }
)
```

## Performance Considerations

### Optimization Strategies
1. **Lazy Evaluation**: Only compute requested statistics
2. **Vectorized Operations**: Use pandas/numpy efficiently
3. **Memory Management**: Process large datasets in chunks if needed
4. **Caching**: Cache expensive computations when appropriate

### Scalability Limits
- Max rows: 1M (inherited from DataLoader)
- Max columns: 1K (inherited from DataLoader)
- Memory usage: Monitor DataFrame memory footprint

## Testing Strategy

### Unit Tests
```python
class TestStatsEngine:
    def test_descriptive_stats_basic()
    def test_descriptive_stats_missing_data()
    def test_correlation_matrix()
    def test_correlation_edge_cases()
    def test_error_handling()
```

### Test Data Scenarios
1. **Clean Data**: No missing values, normal distributions
2. **Missing Data**: Various percentages and patterns
3. **Edge Cases**: Single values, constants, infinities
4. **Performance**: Large datasets (100K+ rows)
5. **Mixed Types**: Numeric and categorical columns

### Integration Tests
- End-to-end workflow with TaskDefinition
- Orchestrator integration
- Error propagation through system

## Implementation Checklist

### Core Implementation
- [ ] StatsEngine class with execute() method
- [ ] calculate_descriptive_stats() implementation
- [ ] calculate_correlation() implementation
- [ ] Missing data handling strategies
- [ ] Comprehensive error handling

### Validation & Quality
- [ ] Input validation for all methods
- [ ] Edge case handling (empty data, single values)
- [ ] Numeric precision and formatting
- [ ] Logging for debugging and monitoring

### Integration
- [ ] TaskDefinition compatibility
- [ ] Exception hierarchy compliance
- [ ] Result format standardization
- [ ] Orchestrator integration points

### Testing
- [ ] Unit tests for all public methods
- [ ] Edge case test coverage
- [ ] Performance testing with large datasets
- [ ] Integration test with full pipeline

## Future Enhancements

### Statistical Operations
- Hypothesis testing (t-tests, chi-square)
- Distribution fitting and analysis
- Outlier detection and analysis
- Time series descriptive statistics

### Advanced Correlations
- Partial correlations
- Correlation significance testing
- Multiple correlation analysis
- Cross-correlation for time series

### Performance Optimizations
- Parallel processing for independent calculations
- Streaming computation for very large datasets
- GPU acceleration for matrix operations
- Incremental statistics for real-time updates