"""
Visualization generator for statistical and ML charts.

Creates matplotlib/seaborn visualizations optimized for Telegram display.
"""

from io import BytesIO
from typing import Dict, Optional, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, roc_curve, auc

from src.processors.dataclasses import ImageData, ProcessorConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VisualizationGenerator:
    """Generates matplotlib/seaborn visualizations optimized for Telegram display."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        self._setup_style()
        logger.info("VisualizationGenerator initialized")

    def _setup_style(self) -> None:
        if self.config.plot_theme == 'dark':
            plt.style.use('dark_background')
            sns.set_palette("husl")
        else:
            plt.style.use('seaborn-v0_8-darkgrid')
            sns.set_palette("Set2")

        # Configure seaborn
        sns.set_context("notebook", font_scale=1.1)

    def generate_histogram(
        self,
        data: pd.DataFrame,
        column: str,
        bins: int = 20
    ) -> ImageData:
        """Generate histogram showing data distribution for specified column."""
        if data.empty:
            raise ValueError("Data cannot be empty")

        if column not in data.columns:
            raise KeyError(f"Column '{column}' not found in data")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot histogram
            data[column].hist(bins=bins, ax=ax, edgecolor='black', alpha=0.7)

            ax.set_xlabel(column, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'Distribution of {column}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = f"📊 Histogram: Distribution of {column}"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_boxplot(
        self,
        data: pd.DataFrame,
        columns: List[str]
    ) -> ImageData:
        """Generate boxplot for outlier detection across multiple columns."""
        if data.empty:
            raise ValueError("Data cannot be empty")

        missing_cols = set(columns) - set(data.columns)
        if missing_cols:
            raise KeyError(f"Columns not found: {missing_cols}")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot boxplot
            data[columns].boxplot(ax=ax)

            ax.set_ylabel('Value', fontsize=12)
            ax.set_title('Box Plot: Outlier Detection', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = f"📦 Box Plot: {', '.join(columns[:3])}"
            if len(columns) > 3:
                caption += f" and {len(columns) - 3} more"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame
    ) -> ImageData:
        """Generate correlation heatmap showing variable relationships."""
        if corr_matrix.empty:
            raise ValueError("Correlation matrix cannot be empty")

        try:
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot heatmap
            sns.heatmap(
                corr_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )

            ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold')
            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = "🔥 Correlation Heatmap: Variable Relationships"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_scatter_plot(
        self,
        x: pd.Series,
        y: pd.Series,
        x_label: str,
        y_label: str,
        hue: Optional[pd.Series] = None
    ) -> ImageData:
        """Generate scatter plot for bivariate relationships with optional hue coloring."""
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Data cannot be empty")

        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot scatter
            if hue is not None:
                scatter = ax.scatter(x, y, c=hue, cmap='viridis', alpha=0.6, s=50)
                plt.colorbar(scatter, ax=ax, label='Category')
            else:
                ax.scatter(x, y, alpha=0.6, s=50, edgecolors='black')

            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            ax.set_title(f'{x_label} vs {y_label}', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = f"📈 Scatter Plot: {x_label} vs {y_label}"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str]
    ) -> ImageData:
        """Generate confusion matrix visualization for classification performance."""
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Data cannot be empty")

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        try:
            # Calculate confusion matrix
            cm = sk_confusion_matrix(y_true, y_pred)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot heatmap
            sns.heatmap(
                cm,
                annot=True,
                fmt='d',
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )

            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = "🎯 Confusion Matrix: Classification Performance"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> ImageData:
        """Generate ROC curve for binary classification with AUC score."""
        if len(y_true) == 0 or len(y_proba) == 0:
            raise ValueError("Data cannot be empty")

        if len(y_true) != len(y_proba):
            raise ValueError("y_true and y_proba must have same length")

        try:
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot ROC curve
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = f"📉 ROC Curve: AUC = {roc_auc:.3f}"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_feature_importance(
        self,
        importances: Dict[str, float],
        top_n: int = 10
    ) -> ImageData:
        """Generate horizontal bar chart showing top N most important features."""
        if not importances:
            raise ValueError("Importances cannot be empty")

        try:
            # Sort by importance and take top N
            sorted_features = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
            top_features = sorted_features[:top_n]

            features, scores = zip(*top_features)

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot horizontal bar chart
            y_pos = np.arange(len(features))
            ax.barh(y_pos, scores, align='center', alpha=0.8, edgecolor='black')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(features)
            ax.invert_yaxis()  # Top feature at top
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = f"⭐ Feature Importance: Top {len(features)} Features"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()

    def generate_residual_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> ImageData:
        """Generate residual plot for regression diagnostics."""
        if len(y_true) == 0 or len(y_pred) == 0:
            raise ValueError("Data cannot be empty")

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have same length")

        try:
            # Calculate residuals
            residuals = y_true - y_pred

            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot residuals
            ax.scatter(y_pred, residuals, alpha=0.6, s=50, edgecolors='black')
            ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

            ax.set_xlabel('Predicted Values', fontsize=12)
            ax.set_ylabel('Residuals', fontsize=12)
            ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=self.config.image_dpi, bbox_inches='tight')
            buffer.seek(0)

            caption = "📐 Residual Plot: Regression Diagnostics"

            return ImageData(buffer=buffer, caption=caption, format='png')

        finally:
            plt.close()
