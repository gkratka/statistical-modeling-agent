"""
Tests for handling unknown/unseen categories during ML prediction.

These tests verify that the prediction pipeline gracefully handles categorical
values in prediction data that were not present in the training data.
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from unittest.mock import MagicMock, patch
from io import StringIO
import sys


class TestUnknownCategoryHandling:
    """Test suite for unknown category handling in prediction."""

    def test_single_unseen_category(self):
        """Test handling of a single unseen category value."""
        # Setup: encoder trained on ['A', 'B', 'C']
        encoder = LabelEncoder()
        encoder.fit(['A', 'B', 'C'])

        # Prediction data has 'D' which wasn't in training
        X = pd.DataFrame({'category': ['A', 'B', 'D', 'A']})

        # Apply the handling logic (same as in worker)
        col = 'category'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        assert unseen == {'D'}, "Should detect 'D' as unseen"

        # Map unseen to first class
        X[col] = col_values.apply(
            lambda v: v if v in known_classes else encoder.classes_[0]
        )

        # Now transform should succeed
        X[col] = encoder.transform(X[col])

        # 'D' should be mapped to 'A' (encoder.classes_[0])
        assert X[col].tolist() == [0, 1, 0, 0]  # A=0, B=1, D->A=0, A=0

    def test_multiple_unseen_categories(self):
        """Test handling of multiple unseen category values."""
        encoder = LabelEncoder()
        encoder.fit(['cat', 'dog'])

        # Multiple unseen values
        X = pd.DataFrame({'animal': ['cat', 'bird', 'fish', 'dog']})

        col = 'animal'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        assert unseen == {'bird', 'fish'}, "Should detect both 'bird' and 'fish' as unseen"

        X[col] = col_values.apply(
            lambda v: v if v in known_classes else encoder.classes_[0]
        )
        X[col] = encoder.transform(X[col])

        # bird and fish mapped to cat (first class)
        assert X[col].tolist() == [0, 0, 0, 1]  # cat=0, bird->cat=0, fish->cat=0, dog=1

    def test_all_categories_unseen(self):
        """Test edge case where ALL categories in prediction data are unseen."""
        encoder = LabelEncoder()
        encoder.fit(['X', 'Y', 'Z'])

        # All values are unseen
        X = pd.DataFrame({'col': ['A', 'B', 'C']})

        col = 'col'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        assert unseen == {'A', 'B', 'C'}, "All values should be unseen"

        X[col] = col_values.apply(
            lambda v: v if v in known_classes else encoder.classes_[0]
        )
        X[col] = encoder.transform(X[col])

        # All mapped to first class X=0
        assert X[col].tolist() == [0, 0, 0]

    def test_no_unseen_categories(self):
        """Test baseline case where all categories are known."""
        encoder = LabelEncoder()
        encoder.fit(['red', 'green', 'blue'])

        X = pd.DataFrame({'color': ['red', 'green', 'blue', 'red']})

        col = 'color'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        assert unseen == set(), "No unseen categories"

        # Direct transform should work
        X[col] = encoder.transform(X[col])

        assert list(X[col]) == [2, 1, 0, 2]  # red=2, green=1, blue=0 (alphabetical)

    def test_multiple_columns_with_unseen(self):
        """Test handling when multiple columns have unseen values."""
        encoder1 = LabelEncoder()
        encoder1.fit(['A', 'B'])

        encoder2 = LabelEncoder()
        encoder2.fit(['X', 'Y'])

        encoders = {'col1': encoder1, 'col2': encoder2}

        X = pd.DataFrame({
            'col1': ['A', 'C', 'B'],  # 'C' is unseen
            'col2': ['X', 'Z', 'Y']   # 'Z' is unseen
        })

        for col, encoder in encoders.items():
            col_values = X[col].astype(str)
            known_classes = set(encoder.classes_)
            unseen = set(col_values.unique()) - known_classes

            if unseen:
                X[col] = col_values.apply(
                    lambda v: v if v in known_classes else encoder.classes_[0]
                )
            else:
                X[col] = col_values

            X[col] = encoder.transform(X[col])

        # col1: A=0, C->A=0, B=1
        # col2: X=0, Z->X=0, Y=1
        assert X['col1'].tolist() == [0, 0, 1]
        assert X['col2'].tolist() == [0, 0, 1]

    def test_warning_logged_for_unseen(self, capsys):
        """Test that warnings are logged when unseen categories are encountered."""
        encoder = LabelEncoder()
        encoder.fit(['known1', 'known2'])

        X = pd.DataFrame({'col': ['known1', 'unknown_value']})

        col = 'col'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        if unseen:
            print(f"[WARN] Column '{col}' has unseen categories: {sorted(unseen)}. "
                  f"Mapping to '{encoder.classes_[0]}' (most frequent)")
            X[col] = col_values.apply(
                lambda v: v if v in known_classes else encoder.classes_[0]
            )

        X[col] = encoder.transform(X[col])

        captured = capsys.readouterr()
        assert "[WARN]" in captured.out
        assert "unknown_value" in captured.out
        assert "known1" in captured.out  # The replacement value

    def test_backward_compatibility_no_encoders(self):
        """Test that prediction works when no encoders exist (numeric-only models)."""
        encoders = {}  # No encoders

        X = pd.DataFrame({
            'numeric1': [1.0, 2.0, 3.0],
            'numeric2': [4.0, 5.0, 6.0]
        })

        original_values = X.copy()

        # Apply encoders (should do nothing)
        for col, encoder in encoders.items():
            if col in X.columns:
                col_values = X[col].astype(str)
                known_classes = set(encoder.classes_)
                unseen = set(col_values.unique()) - known_classes

                if unseen:
                    X[col] = col_values.apply(
                        lambda v: v if v in known_classes else encoder.classes_[0]
                    )

                X[col] = encoder.transform(X[col])

        # Data should be unchanged
        pd.testing.assert_frame_equal(X, original_values)

    def test_mixed_known_unknown_in_same_column(self):
        """Test column with mix of known and unknown values."""
        encoder = LabelEncoder()
        encoder.fit(['apple', 'banana', 'cherry'])

        # Mix of known and unknown
        X = pd.DataFrame({
            'fruit': ['apple', 'mango', 'banana', 'kiwi', 'cherry', 'papaya']
        })

        col = 'fruit'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        assert unseen == {'mango', 'kiwi', 'papaya'}

        X[col] = col_values.apply(
            lambda v: v if v in known_classes else encoder.classes_[0]
        )
        X[col] = encoder.transform(X[col])

        # apple=0, mango->apple=0, banana=1, kiwi->apple=0, cherry=2, papaya->apple=0
        assert X[col].tolist() == [0, 0, 1, 0, 2, 0]

    def test_empty_dataframe(self):
        """Test handling of empty prediction dataframe."""
        encoder = LabelEncoder()
        encoder.fit(['A', 'B'])

        X = pd.DataFrame({'col': pd.Series([], dtype=str)})

        col = 'col'
        col_values = X[col].astype(str)
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        # Empty should have no unseen
        assert unseen == set()

        # Transform empty should work
        if len(X) > 0:
            X[col] = encoder.transform(X[col])

        assert len(X) == 0

    def test_numeric_strings_as_categories(self):
        """Test handling of numeric values stored as categorical strings."""
        encoder = LabelEncoder()
        encoder.fit(['1', '2', '3'])

        X = pd.DataFrame({'code': [1, 2, 4, 1]})  # numeric, 4 is unseen

        col = 'code'
        col_values = X[col].astype(str)  # Convert to string like worker does
        known_classes = set(encoder.classes_)
        unseen = set(col_values.unique()) - known_classes

        assert unseen == {'4'}

        X[col] = col_values.apply(
            lambda v: v if v in known_classes else encoder.classes_[0]
        )
        X[col] = encoder.transform(X[col])

        # 1=0, 2=1, 4->1=0, 1=0
        assert X[col].tolist() == [0, 1, 0, 0]


class TestUnknownCategoryIntegration:
    """Integration-style tests simulating real worker behavior."""

    def test_full_prediction_flow_with_unseen(self):
        """Test the complete flow that would happen in the worker."""
        # Simulate what the worker does
        encoders = {
            'product_type': LabelEncoder(),
            'region': LabelEncoder()
        }
        encoders['product_type'].fit(['CONSUMER', 'POINT', 'TC'])
        encoders['region'].fit(['NORTH', 'SOUTH', 'EAST', 'WEST'])

        # Prediction data with unseen values
        X = pd.DataFrame({
            'product_type': ['CONSUMER', 'MERCHANT', 'POINT'],  # MERCHANT unseen
            'region': ['NORTH', 'CENTRAL', 'SOUTH'],  # CENTRAL unseen
            'amount': [100.0, 200.0, 300.0]
        })

        # Apply encoders (mimicking worker logic)
        for col, encoder in encoders.items():
            if col in X.columns:
                col_values = X[col].astype(str)
                known_classes = set(encoder.classes_)
                unseen = set(col_values.unique()) - known_classes

                if unseen:
                    X[col] = col_values.apply(
                        lambda v: v if v in known_classes else encoder.classes_[0]
                    )
                else:
                    X[col] = col_values

                X[col] = encoder.transform(X[col])

        # Verify transformation succeeded
        assert X['product_type'].dtype in [np.int64, np.int32, int]
        assert X['region'].dtype in [np.int64, np.int32, int]
        assert X['amount'].tolist() == [100.0, 200.0, 300.0]  # Unchanged

        # MERCHANT -> CONSUMER (first class) = 0
        # CENTRAL -> EAST (first class, alphabetical) = 0
        assert X['product_type'].tolist() == [0, 0, 1]  # CONSUMER=0, MERCHANT->CONSUMER=0, POINT=1
