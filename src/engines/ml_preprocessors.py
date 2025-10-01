"""
ML Data Preprocessing Utilities.

This module provides data preprocessing functions for ML operations including
missing value handling, feature scaling, and categorical encoding.
"""

from typing import Any, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

from src.utils.exceptions import ValidationError


class MLPreprocessors:
    """Data preprocessing utilities for ML operations."""

    @staticmethod
    def handle_missing_values(
        X: pd.DataFrame,
        strategy: str = "mean",
        fill_value: Any = None
    ) -> pd.DataFrame:
        """
        Handle missing values in features.

        Args:
            X: Feature dataframe
            strategy: Strategy for handling missing values
                     ("mean", "median", "drop", "zero", "constant")
            fill_value: Value to use for "constant" strategy

        Returns:
            DataFrame with missing values handled

        Raises:
            ValidationError: If strategy is unknown
        """
        X = X.copy()

        if strategy == "mean":
            # Fill with column means (numeric columns only)
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].mean())

        elif strategy == "median":
            # Fill with column medians (numeric columns only)
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())

        elif strategy == "drop":
            # Drop rows with any missing values
            X = X.dropna()

        elif strategy == "zero":
            # Fill with zeros
            X = X.fillna(0)

        elif strategy == "constant":
            # Fill with specified constant value
            if fill_value is None:
                raise ValidationError(
                    "fill_value must be provided for 'constant' strategy",
                    field="fill_value"
                )
            X = X.fillna(fill_value)

        else:
            raise ValidationError(
                f"Unknown missing value strategy: '{strategy}'. "
                "Must be one of: mean, median, drop, zero, constant",
                field="missing_strategy",
                value=strategy
            )

        return X

    @staticmethod
    def scale_features(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        method: str = "standard"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Any]:
        """
        Scale numerical features.

        Args:
            X_train: Training features
            X_test: Test features
            method: Scaling method ("standard", "minmax", "robust", "none")

        Returns:
            Tuple of (X_train_scaled, X_test_scaled, scaler_object)

        Raises:
            ValidationError: If method is unknown
        """
        if method == "none":
            return X_train, X_test, None

        # Select scaler
        scalers = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }

        if method not in scalers:
            raise ValidationError(
                f"Unknown scaling method: '{method}'. "
                "Must be one of: standard, minmax, robust, none",
                field="scaling",
                value=method
            )

        scaler = scalers[method]

        # Fit on training data and transform both sets
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_test_scaled, scaler

    @staticmethod
    def encode_categorical(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
        """
        Encode categorical variables using label encoding.

        Args:
            X_train: Training features
            X_test: Test features

        Returns:
            Tuple of (X_train_encoded, X_test_encoded, encoders_dict)
        """
        encoders = {}
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        # Identify categorical columns
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns

        for col in categorical_cols:
            encoder = LabelEncoder()

            # Fit on training data
            X_train_encoded[col] = encoder.fit_transform(X_train[col].astype(str))

            # Transform test data, handling unseen categories
            try:
                X_test_encoded[col] = encoder.transform(X_test[col].astype(str))
            except ValueError:
                # Handle unseen categories by assigning them to a new class
                # Store known classes
                known_classes = set(encoder.classes_)

                # Create mapping for test data
                test_encoded = []
                for val in X_test[col].astype(str):
                    if val in known_classes:
                        test_encoded.append(encoder.transform([val])[0])
                    else:
                        # Assign unseen category to -1 or max+1
                        test_encoded.append(len(encoder.classes_))

                X_test_encoded[col] = test_encoded

            encoders[col] = encoder

        return X_train_encoded, X_test_encoded, encoders

    @staticmethod
    def detect_outliers_iqr(
        X: pd.DataFrame,
        column: str,
        multiplier: float = 1.5
    ) -> pd.Series:
        """
        Detect outliers using IQR method.

        Args:
            X: Feature dataframe
            column: Column name to check for outliers
            multiplier: IQR multiplier for outlier threshold (default 1.5)

        Returns:
            Boolean series indicating outliers (True = outlier)
        """
        Q1 = X[column].quantile(0.25)
        Q3 = X[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        return (X[column] < lower_bound) | (X[column] > upper_bound)

    @staticmethod
    def remove_outliers(
        X: pd.DataFrame,
        y: pd.Series,
        columns: list[str] = None,
        multiplier: float = 1.5
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Remove outliers from dataset using IQR method.

        Args:
            X: Feature dataframe
            y: Target series
            columns: List of columns to check (default: all numeric)
            multiplier: IQR multiplier for outlier threshold

        Returns:
            Tuple of (X_cleaned, y_cleaned) with outliers removed
        """
        if columns is None:
            columns = X.select_dtypes(include=[np.number]).columns.tolist()

        # Create mask for non-outliers
        mask = pd.Series([True] * len(X), index=X.index)

        for col in columns:
            if col in X.columns:
                outliers = MLPreprocessors.detect_outliers_iqr(X, col, multiplier)
                mask = mask & ~outliers

        # Apply mask
        X_cleaned = X[mask].copy()
        y_cleaned = y[mask].copy()

        return X_cleaned, y_cleaned

    @staticmethod
    def balance_classes(
        X: pd.DataFrame,
        y: pd.Series,
        strategy: str = "undersample"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Balance classes for classification problems.

        Args:
            X: Feature dataframe
            y: Target series
            strategy: Balancing strategy ("undersample" or "oversample")

        Returns:
            Tuple of (X_balanced, y_balanced)

        Raises:
            ValidationError: If strategy is unknown
        """
        if strategy not in ["undersample", "oversample"]:
            raise ValidationError(
                f"Unknown balancing strategy: '{strategy}'. "
                "Must be 'undersample' or 'oversample'",
                field="balance_strategy",
                value=strategy
            )

        # Get class counts
        class_counts = y.value_counts()

        if strategy == "undersample":
            # Downsample to size of minority class
            min_count = class_counts.min()

            # Sample from each class
            balanced_indices = []
            for class_label in class_counts.index:
                class_indices = y[y == class_label].index
                sampled_indices = np.random.choice(
                    class_indices,
                    size=min_count,
                    replace=False
                )
                balanced_indices.extend(sampled_indices)

        else:  # oversample
            # Upsample to size of majority class
            max_count = class_counts.max()

            # Sample from each class
            balanced_indices = []
            for class_label in class_counts.index:
                class_indices = y[y == class_label].index
                sampled_indices = np.random.choice(
                    class_indices,
                    size=max_count,
                    replace=True
                )
                balanced_indices.extend(sampled_indices)

        # Shuffle indices
        np.random.shuffle(balanced_indices)

        # Return balanced data
        X_balanced = X.loc[balanced_indices].copy()
        y_balanced = y.loc[balanced_indices].copy()

        return X_balanced, y_balanced

    @staticmethod
    def get_preprocessing_pipeline_info(
        scaling: str = None,
        missing_strategy: str = None,
        encode_categorical: bool = False
    ) -> dict:
        """
        Get information about preprocessing pipeline configuration.

        Args:
            scaling: Scaling method used
            missing_strategy: Missing value strategy used
            encode_categorical: Whether categorical encoding was applied

        Returns:
            Dictionary with preprocessing configuration
        """
        return {
            "scaled": scaling is not None and scaling != "none",
            "scaler_type": scaling if scaling and scaling != "none" else None,
            "missing_value_strategy": missing_strategy,
            "categorical_encoded": encode_categorical
        }
