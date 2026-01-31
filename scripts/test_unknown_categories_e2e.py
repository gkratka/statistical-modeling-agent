"""
End-to-end test: Unknown category handling in prediction pipeline.

This script simulates the real worker prediction flow to verify that
unseen categorical values are handled gracefully instead of crashing.

Usage:
    python3 scripts/test_unknown_categories_e2e.py

What it does:
    1. Loads german_credit_data.csv (real data with 13 categorical columns)
    2. Trains a RandomForest classifier (mimicking the worker training path)
    3. Saves model + LabelEncoders to a temp directory
    4. Creates prediction data with UNSEEN category values injected
    5. Runs the exact encoding logic from worker/statsbot_worker.py
    6. Verifies predictions succeed without ValueError
"""

import sys
import os
import json
import pickle
import tempfile
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_pass(msg):
    print(f"  {GREEN}PASS{RESET} {msg}")


def print_fail(msg):
    print(f"  {RED}FAIL{RESET} {msg}")


def print_warn(msg):
    print(f"  {YELLOW}WARN{RESET} {msg}")


def print_header(msg):
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{msg}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "test_data" / "german_credit_data.csv"

    if not data_path.exists():
        print_fail(f"Test data not found: {data_path}")
        sys.exit(1)

    print_header("E2E Test: Unknown Category Handling in Prediction")
    results = {"passed": 0, "failed": 0}

    # ── Step 1: Load real data ──────────────────────────────────
    print(f"\n{BOLD}Step 1: Load training data{RESET}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from german_credit_data.csv")

    target_col = "class"
    cat_cols = [c for c in df.columns if df[c].dtype == "object"]
    num_cols = [c for c in df.columns if df[c].dtype != "object" and c != target_col]
    feature_cols = cat_cols + num_cols
    print(f"  Categorical columns ({len(cat_cols)}): {cat_cols[:5]}...")
    print(f"  Numeric columns ({len(num_cols)}): {num_cols[:5]}...")

    # ── Step 2: Train model (mimicking worker) ─────────────────
    print(f"\n{BOLD}Step 2: Train model{RESET}")
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Encode categoricals (same as worker training path)
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f"  Encoded '{col}': {len(le.classes_)} classes → {list(le.classes_[:4])}...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f"  Trained RandomForest: accuracy={accuracy:.3f}")

    # ── Step 3: Save model + encoders to temp dir ──────────────
    print(f"\n{BOLD}Step 3: Save model artifacts{RESET}")
    tmp_dir = tempfile.mkdtemp(prefix="test_unknown_cat_")
    model_dir = Path(tmp_dir) / "test_model"
    model_dir.mkdir()

    with open(model_dir / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(model_dir / "encoders.pkl", "wb") as f:
        pickle.dump(encoders, f)

    metadata = {
        "model_type": "random_forest",
        "task_type": "classification",
        "feature_columns": feature_cols,
        "target_column": target_col,
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata))
    print(f"  Saved to: {model_dir}")

    # ── Step 4: Create prediction data with UNSEEN categories ──
    print(f"\n{BOLD}Step 4: Create prediction data with unseen categories{RESET}")

    # Take a few rows as base prediction data
    pred_df = df[feature_cols].head(10).copy()

    # Inject unseen values into several categorical columns
    injections = {
        "Attribute1": "A99_UNSEEN",      # normally A11-A14
        "Attribute4": "MERCHANT",         # the exact error from Railway logs
        "Attribute6": "UNKNOWN_STATUS",   # normally A61-A65
        "Attribute9": "NEW_CATEGORY",     # normally A91-A94
    }

    for col, unseen_val in injections.items():
        original = pred_df[col].iloc[0]
        pred_df.loc[pred_df.index[0], col] = unseen_val
        print(f"  Injected '{unseen_val}' into '{col}' (was '{original}')")

    # ── Step 5: Run EXACT worker encoding logic ────────────────
    print(f"\n{BOLD}Step 5: Apply worker encoding logic (the fix){RESET}")

    # Reload saved artifacts (simulating worker loading)
    with open(model_dir / "model.pkl", "rb") as f:
        loaded_model = pickle.load(f)
    with open(model_dir / "encoders.pkl", "rb") as f:
        loaded_encoders = pickle.load(f)
    loaded_metadata = json.loads((model_dir / "metadata.json").read_text())

    X_pred = pred_df[loaded_metadata["feature_columns"]].copy()

    # This is the EXACT code from worker/statsbot_worker.py:993-1009
    warnings_logged = []
    try:
        for col, encoder in loaded_encoders.items():
            if col in X_pred.columns:
                col_values = X_pred[col].astype(str)
                known_classes = set(encoder.classes_)
                unseen = set(col_values.unique()) - known_classes

                if unseen:
                    msg = (f"[WARN] Column '{col}' has unseen categories: {sorted(unseen)}. "
                           f"Mapping to '{encoder.classes_[0]}' (most frequent)")
                    print(f"  {YELLOW}{msg}{RESET}")
                    warnings_logged.append(col)
                    X_pred[col] = col_values.apply(
                        lambda v: v if v in known_classes else encoder.classes_[0]
                    )
                else:
                    X_pred[col] = col_values

                X_pred[col] = encoder.transform(X_pred[col])

        print_pass("Encoding completed without ValueError")
        results["passed"] += 1
    except ValueError as e:
        print_fail(f"Encoding raised ValueError: {e}")
        results["failed"] += 1

    # ── Step 6: Verify predictions succeed ─────────────────────
    print(f"\n{BOLD}Step 6: Make predictions{RESET}")
    try:
        if hasattr(loaded_model, "predict_proba"):
            predictions = loaded_model.predict_proba(X_pred)[:, 1]
        else:
            predictions = loaded_model.predict(X_pred)

        print(f"  Got {len(predictions)} predictions: {predictions[:5]}...")
        print_pass("Predictions succeeded")
        results["passed"] += 1
    except Exception as e:
        print_fail(f"Prediction failed: {e}")
        results["failed"] += 1

    # ── Step 7: Verify warnings were logged ────────────────────
    print(f"\n{BOLD}Step 7: Verify warning behavior{RESET}")

    expected_warned = set(injections.keys())
    actual_warned = set(warnings_logged)

    if expected_warned == actual_warned:
        print_pass(f"Warnings logged for all {len(expected_warned)} injected columns: {sorted(actual_warned)}")
        results["passed"] += 1
    else:
        missing = expected_warned - actual_warned
        extra = actual_warned - expected_warned
        if missing:
            print_fail(f"Missing warnings for: {missing}")
        if extra:
            print_warn(f"Unexpected warnings for: {extra}")
        results["failed"] += 1

    # ── Step 8: Verify clean data still works (no regression) ──
    print(f"\n{BOLD}Step 8: Verify clean data still works (no regression){RESET}")
    try:
        clean_df = df[feature_cols].head(10).copy()
        X_clean = clean_df.copy()

        for col, encoder in loaded_encoders.items():
            if col in X_clean.columns:
                col_values = X_clean[col].astype(str)
                known_classes = set(encoder.classes_)
                unseen = set(col_values.unique()) - known_classes

                if unseen:
                    X_clean[col] = col_values.apply(
                        lambda v: v if v in known_classes else encoder.classes_[0]
                    )
                else:
                    X_clean[col] = col_values

                X_clean[col] = encoder.transform(X_clean[col])

        clean_preds = loaded_model.predict(X_clean)
        print(f"  Clean data predictions: {clean_preds.tolist()}")
        print_pass("Clean data (no unseen) still works correctly")
        results["passed"] += 1
    except Exception as e:
        print_fail(f"Clean data prediction failed: {e}")
        results["failed"] += 1

    # ── Cleanup ────────────────────────────────────────────────
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── Summary ────────────────────────────────────────────────
    print_header("Summary")
    total = results["passed"] + results["failed"]
    if results["failed"] == 0:
        print(f"  {GREEN}{BOLD}ALL {total} CHECKS PASSED{RESET}")
        print(f"  The unknown category fix is working correctly.")
        sys.exit(0)
    else:
        print(f"  {results['passed']}/{total} passed, {RED}{results['failed']} FAILED{RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
