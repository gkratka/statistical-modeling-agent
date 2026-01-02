"""
Main result processor orchestrating all processing components.

Coordinates visualization generation, language summaries, and pagination
to produce user-friendly Telegram outputs.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import pandas as pd
import numpy as np

from src.processors.dataclasses import (
    ProcessedResult,
    ImageData,
    ProcessorConfig,
    PaginationState
)
from src.processors.visualization_generator import VisualizationGenerator
from src.processors.language_generator import LanguageGenerator
from src.processors.pagination_manager import PaginationManager
from src.utils.result_formatter import TelegramResultFormatter
from src.utils.logger import get_logger
from src.bot.messages.training_messages import format_training_metrics

logger = get_logger(__name__)


class ResultProcessor:
    """Main orchestrator coordinating visualizations, summaries, and pagination for Telegram output."""

    def __init__(self, config: Optional[ProcessorConfig] = None):
        self.config = config or ProcessorConfig()
        self.text_formatter = TelegramResultFormatter(
            precision=4,
            use_emojis=self.config.use_emojis,
            compact_mode=(self.config.detail_level == "compact")
        )

        # Initialize processing components
        if self.config.enable_visualizations:
            self.viz_generator = VisualizationGenerator(self.config)
        else:
            self.viz_generator = None

        self.lang_generator = LanguageGenerator(self.config)
        self.pagination_manager = PaginationManager(self.config)

        logger.info(f"ResultProcessor initialized with config: {self.config}")

    def _apply_pagination(self, text: str, images: List[ImageData]) -> Tuple[str, bool, Optional[PaginationState]]:
        """Apply pagination if needed, return modified text, needs_pagination flag, and state."""
        needs_pagination = self.pagination_manager.should_paginate(text=text, n_images=len(images))

        if not needs_pagination:
            return text, False, None

        result_id = self.pagination_manager.generate_result_id()
        text_chunks = self.pagination_manager.chunk_text(text)
        pagination_state = self.pagination_manager.create_pagination_state(
            result_id=result_id,
            total_chunks=len(text_chunks),
            current_page=1
        )
        # Return first page with headers/footers
        text = text_chunks[0]
        text = self.pagination_manager.get_page_header(pagination_state) + text
        text += self.pagination_manager.get_page_footer(pagination_state)

        return text, True, pagination_state

    def process_result(
        self,
        result: Dict[str, Any],
        result_type: str
    ) -> ProcessedResult:
        """Process raw result into user-friendly output with visualizations and summaries."""
        logger.info(f"Processing {result_type} result")

        # Route to appropriate processor
        if result_type == "stats":
            return self._process_stats_result(result)
        elif result_type == "ml_training":
            return self._process_ml_training_result(result)
        elif result_type == "ml_prediction":
            return self._process_ml_prediction_result(result)
        else:
            raise ValueError(f"Unknown result type: {result_type}")

    def _process_stats_result(self, result: Dict[str, Any]) -> ProcessedResult:
        """Process statistical analysis results with formatted text and visualizations."""
        # Use existing formatter for text
        text = self.text_formatter.format_stats_result(result)

        # Generate plain language summary
        operation = result.get("operation", "")
        summary = self._generate_stats_summary(result, operation)

        # Generate visualizations if enabled
        images: List[ImageData] = []
        if self.viz_generator and self.config.enable_visualizations:
            images = self._generate_stats_visualizations(result, operation)

        # Apply pagination if needed
        text, needs_pagination, pagination_state = self._apply_pagination(text, images)

        return ProcessedResult(
            text=text,
            images=images,
            files=[],
            summary=summary,
            needs_pagination=needs_pagination,
            pagination_state=pagination_state
        )

    def _generate_stats_summary(self, result: Dict[str, Any], operation: str) -> str:
        """Generate plain language summary for stats results."""
        try:
            if operation == "descriptive":
                stats = result.get("statistics", {})
                return self.lang_generator.generate_descriptive_stats_summary(stats)
            elif operation == "correlation":
                corr = result.get("correlation_matrix", {})
                # Extract key correlations
                correlations = {
                    f"{key}_{k2}": v
                    for key, values in corr.items()
                    if isinstance(values, dict)
                    for k2, v in values.items()
                    if key != k2
                }
                return self.lang_generator.generate_correlation_summary(correlations)
            elif operation == "regression":
                return self.lang_generator.generate_regression_summary(result)
            else:
                return "Statistical analysis completed successfully."
        except Exception as e:
            logger.warning(f"Could not generate language summary: {e}")
            return "Statistical analysis complete."

    def _generate_stats_visualizations(
        self,
        result: Dict[str, Any],
        operation: str
    ) -> List[ImageData]:
        """Generate visualizations for stats results."""
        images = []
        data = result.get("data")

        if not isinstance(data, pd.DataFrame):
            return images

        try:
            if operation == "descriptive":
                # Generate histogram for each numeric column
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                for col in numeric_cols[:self.config.max_charts_per_result]:
                    img = self.viz_generator.generate_histogram(data, col)
                    images.append(img)

            elif operation == "correlation":
                # Generate correlation heatmap
                numeric_data = data.select_dtypes(include=[np.number])
                if not numeric_data.empty:
                    corr_matrix = numeric_data.corr()
                    img = self.viz_generator.generate_correlation_heatmap(corr_matrix)
                    images.append(img)

            elif operation == "distribution":
                # Generate boxplots
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    img = self.viz_generator.generate_boxplot(data, numeric_cols[:5])
                    images.append(img)

        except Exception as e:
            logger.warning(f"Could not generate visualizations: {e}")

        return images

    def _process_ml_training_result(self, result: Dict[str, Any]) -> ProcessedResult:
        """Process ML training results with formatted metrics and charts."""
        # Check if this is an error result first
        if result.get("success") is False:
            # Handle error case - format error message
            error_msg = result.get("error", "Unknown error")
            error_code = result.get("error_code", "ERROR")
            user_message = result.get("message", "")
            suggestions = result.get("suggestions", [])

            emoji = "❌ " if self.config.use_emojis else ""
            text = f"{emoji}Training Failed\n\n"
            text += f"Error: {error_msg}\n\n"

            if suggestions:
                text += "Suggestions:\n"
                for i, suggestion in enumerate(suggestions, 1):
                    text += f"{i}. {suggestion}\n"

            return ProcessedResult(
                text=text,
                images=[],
                files=[],
                summary=f"Training failed: {error_msg}",
                needs_pagination=False,
                pagination_state=None
            )

        # Success case - process training results
        model_id = result.get("model_id", "unknown")
        metrics = result.get("metrics", {})
        model_info = result.get("model_info", {})
        model_type = model_info.get("model_type", "unknown")
        task_type = model_info.get("task_type", "classification")

        # Build text output with model ID and metrics
        emoji = "🤖 " if self.config.use_emojis else ""
        text = f"{emoji}**ML Training Complete**\n\n"
        text += f"🎯 Model: {model_type}\n"
        text += f"Model ID: `{model_id}`\n\n"
        text += "**📈 Performance Metrics:**\n"

        # Use priority-ordered metrics formatter
        text += format_training_metrics(metrics, task_type) + "\n"

        # Add regression equation if available (for simple linear models)
        model_info = result.get('model_info', {})
        coefficients = model_info.get('coefficients', {})
        intercept = model_info.get('intercept')
        target = model_info.get('target')

        if coefficients and intercept is not None and len(coefficients) == 1:
            # Simple linear regression: y = a*x + b
            feature_name = list(coefficients.keys())[0]
            coef_value = list(coefficients.values())[0]

            text += f"\n**Regression Equation:**\n"
            target_display = target if target else "y"
            text += f"• {target_display} = {coef_value:.2f} * {feature_name} + {intercept:.2f}\n"

        # Generate plain language summary
        summary = self.lang_generator.generate_ml_training_summary(result)

        # Generate visualizations if enabled
        images: List[ImageData] = []
        if self.viz_generator and self.config.enable_visualizations:
            images = self._generate_ml_training_visualizations(result)

        # Apply pagination if needed
        text, needs_pagination, pagination_state = self._apply_pagination(text, images)

        return ProcessedResult(
            text=text,
            images=images,
            files=[],
            summary=summary,
            needs_pagination=needs_pagination,
            pagination_state=pagination_state
        )

    def _generate_ml_training_visualizations(
        self,
        result: Dict[str, Any]
    ) -> List[ImageData]:
        """Generate visualizations for ML training results."""
        images = []

        try:
            # Confusion matrix for classification
            if "confusion_matrix" in result and "labels" in result:
                y_true = result.get("y_true")
                y_pred = result.get("y_pred")
                labels = result.get("labels")

                if y_true is not None and y_pred is not None:
                    img = self.viz_generator.generate_confusion_matrix(
                        y_true=np.array(y_true),
                        y_pred=np.array(y_pred),
                        labels=labels
                    )
                    images.append(img)

            # ROC curve for binary classification
            if "roc_curve" in result:
                y_true = result.get("y_true")
                y_proba = result.get("y_proba")

                if y_true is not None and y_proba is not None:
                    img = self.viz_generator.generate_roc_curve(
                        y_true=np.array(y_true),
                        y_proba=np.array(y_proba)
                    )
                    images.append(img)

            # Feature importance
            if "feature_importance" in result:
                importances = result["feature_importance"]
                if importances:
                    img = self.viz_generator.generate_feature_importance(importances)
                    images.append(img)

        except Exception as e:
            logger.warning(f"Could not generate ML training visualizations: {e}")

        return images

    def _process_ml_prediction_result(self, result: Dict[str, Any]) -> ProcessedResult:
        """Process ML prediction results with predictions and statistics."""
        predictions = result.get("predictions", [])
        n_predictions = result.get("n_predictions", len(predictions))

        # Build text output
        emoji = "🔮 " if self.config.use_emojis else ""
        text = f"{emoji}**ML Predictions Complete**\n\n"
        text += f"Made {n_predictions} predictions\n\n"

        # Show sample predictions
        if predictions and len(predictions) <= 10:
            text += "**First predictions:**\n"
            for i, pred in enumerate(predictions[:10], 1):
                text += f"{i}. {pred}\n"

        # Add prediction statistics if available
        if "prediction_summary" in result:
            pred_stats = result["prediction_summary"]
            text += "\n**Prediction Statistics:**\n"
            for key, value in pred_stats.items():
                if isinstance(value, (int, float)):
                    text += f"• {key}: {value:.2f}\n"

        # Generate plain language summary
        summary = self.lang_generator.generate_ml_prediction_summary(result)

        # Generate visualizations if available
        images: List[ImageData] = []
        if self.viz_generator and self.config.enable_visualizations:
            images = self._generate_ml_prediction_visualizations(result)

        # Apply pagination if needed
        text, needs_pagination, pagination_state = self._apply_pagination(text, images)

        return ProcessedResult(
            text=text,
            images=images,
            files=[],
            summary=summary,
            needs_pagination=needs_pagination,
            pagination_state=pagination_state
        )

    def _generate_ml_prediction_visualizations(
        self,
        result: Dict[str, Any]
    ) -> List[ImageData]:
        """Generate visualizations for ML prediction results."""
        images = []

        try:
            # Histogram of predictions distribution
            predictions = result.get("predictions")
            if predictions and len(predictions) > 10:
                pred_df = pd.DataFrame({"predictions": predictions})
                img = self.viz_generator.generate_histogram(
                    pred_df,
                    "predictions",
                    bins=30
                )
                images.append(img)

            # Residual plot if true values available (for regression)
            if "y_true" in result and "predictions" in result:
                y_true = np.array(result["y_true"])
                y_pred = np.array(result["predictions"])

                if len(y_true) == len(y_pred):
                    img = self.viz_generator.generate_residual_plot(y_true, y_pred)
                    images.append(img)

        except Exception as e:
            logger.warning(f"Could not generate ML prediction visualizations: {e}")

        return images

    def get_supported_result_types(self) -> List[str]:
        """Get list of supported result types."""
        return ["stats", "ml_training", "ml_prediction"]
