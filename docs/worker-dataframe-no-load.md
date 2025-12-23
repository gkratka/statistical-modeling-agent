# Worker Dataframe Memory Optimization Plan

## Problem Statement

The Railway deployment crashes with OOM (Out of Memory) errors when processing prediction results from the local worker. The bot receives a 1-10GB dataframe via WebSocket, causing memory exhaustion.

### Root Cause

In `worker/statsbot_worker.py` (lines 911-919), the worker sends the **entire dataframe** back to the bot:

```python
return create_result_message(
    job_id,
    True,
    data={
        "predictions": predictions.tolist(),
        "count": len(predictions),
        "dataframe": df.to_dict('records'),  # ← PROBLEM: Full dataframe (1-10GB)
    },
)
```

The bot (Railway, 512MB-2GB RAM) cannot handle receiving multi-GB payloads.

## Solution

### Strategy: Truncate + File Output

1. **Worker**: Send only a sample (first 10 rows) for display
2. **Worker**: Save full results to CSV file on local machine
3. **Bot**: Display sample + file path to user

### Implementation Details

#### 1. Modify Worker Result Payload (`worker/statsbot_worker.py`)

**Before:**
```python
data={
    "predictions": predictions.tolist(),
    "count": len(predictions),
    "dataframe": df.to_dict('records'),
}
```

**After:**
```python
# Save full results to file
output_path = f"predictions_{job_id}.csv"
df_with_predictions = df.copy()
df_with_predictions['prediction'] = predictions
df_with_predictions.to_csv(output_path, index=False)

# Send only sample
SAMPLE_SIZE = 10
data={
    "predictions_sample": predictions[:SAMPLE_SIZE].tolist(),
    "predictions_count": len(predictions),
    "dataframe_sample": df.head(SAMPLE_SIZE).to_dict('records'),
    "dataframe_rows": len(df),
    "dataframe_columns": list(df.columns),
    "output_file": output_path,
}
```

#### 2. Modify Bot Result Handler (`src/bot/ml_handlers/prediction_handlers.py`)

Update to handle truncated results:
- Display sample rows with predictions
- Show total count
- Inform user where full results are saved

#### 3. Add Tests (`tests/unit/test_worker_dataframe.py`)

```python
def test_result_payload_is_truncated():
    """Verify worker doesn't send full dataframe."""
    # Create large dataframe
    # Execute prediction
    # Assert result['dataframe_sample'] has max 10 rows
    # Assert 'output_file' path is included

def test_full_results_saved_to_file():
    """Verify full predictions saved to CSV."""
    # Execute prediction
    # Assert file exists at output_file path
    # Assert file contains all rows
```

## File Changes Summary

| File | Change |
|------|--------|
| `worker/statsbot_worker.py` | Truncate dataframe, save to file |
| `src/bot/ml_handlers/prediction_handlers.py` | Handle truncated results |
| `tests/unit/test_worker_dataframe.py` | New test file |

## Branch Strategy

1. This document committed to `main` branch
2. Implementation in `feature/worker-no-load-dataframe` branch
3. PR and merge when complete

## Success Criteria

- [ ] Worker result payload < 1MB regardless of input size
- [ ] Full predictions saved to local CSV file
- [ ] Bot displays sample + file location
- [ ] No OOM crashes on Railway for large predictions
- [ ] All tests pass

## Risk Mitigation

- **File path access**: User's local worker saves file locally; path is informational only
- **Backwards compatibility**: Check for both old (`dataframe`) and new (`dataframe_sample`) keys in bot handler
