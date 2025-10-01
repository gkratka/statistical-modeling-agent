"""
Language generator for plain English statistical summaries.

Converts technical statistical and ML results into user-friendly
natural language explanations.
"""

from typing import Dict, Any, List
import numpy as np

from src.processors.dataclasses import ProcessorConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LanguageGenerator:
    """Converts technical metrics into plain language summaries tailored to user preferences."""

    def __init__(self, config: ProcessorConfig):
        self.config = config
        logger.info(f"LanguageGenerator initialized with style: {config.language_style}")

    def _emoji(self, char: str) -> str:
        """Get emoji character if enabled in config."""
        return f"{char} " if self.config.use_emojis else ""

    def generate_descriptive_stats_summary(self, stats: Dict[str, Dict[str, float]]) -> str:
        """Generate plain language summary of descriptive statistics."""
        if not stats:
            raise ValueError("Statistics dictionary cannot be empty")

        parts = []

        if self.config.language_style == "technical":
            parts.append(f"{self._emoji('📊')}Statistical Summary:")
            for var, metrics in stats.items():
                mean = metrics.get('mean', 0)
                std = metrics.get('std', 0)
                parts.append(f"• {var}: mean={mean:.2f}, std={std:.2f}")
        else:
            parts.append(f"{self._emoji('📊')}Here's what the data shows:")
            for var, metrics in stats.items():
                mean = metrics.get('mean')
                median = metrics.get('median')
                std = metrics.get('std')

                if mean is not None:
                    parts.append(f"\n**{var}:**")
                    parts.append(f"• The average value is {mean:.2f}")

                    if median is not None:
                        if abs(mean - median) < 0.1 * mean:
                            parts.append(f"• The typical value is {median:.2f} (similar to average)")
                        else:
                            parts.append(f"• The typical value is {median:.2f} (differs from average)")

                    if std is not None:
                        cv = (std / mean) * 100 if mean != 0 else 0
                        if cv < 10:
                            parts.append(f"• Values are very consistent (low spread)")
                        elif cv < 30:
                            parts.append(f"• Values show moderate variation")
                        else:
                            parts.append(f"• Values vary quite a bit (high spread)")

        return "\n".join(parts)

    def generate_correlation_summary(self, correlations: Dict[str, float]) -> str:
        """Generate plain language summary of correlation relationships."""
        if not correlations:
            raise ValueError("Correlations dictionary cannot be empty")

        parts = []

        if self.config.language_style == "technical":
            parts.append(f"{self._emoji('🔗')}Correlation Analysis:")
            for pair, corr in correlations.items():
                parts.append(f"• {pair}: r={corr:.3f} ({self.interpret_correlation_strength(corr)})")
        else:
            parts.append(f"{self._emoji('🔗')}Relationships between variables:")

            # Find strongest correlation
            strongest_pair = max(correlations.items(), key=lambda x: abs(x[1]))
            parts.append(f"\n**Strongest relationship:** {strongest_pair[0]}")
            parts.append(f"• {self.interpret_correlation_strength(strongest_pair[1])}")

            # Summarize others
            if len(correlations) > 1:
                parts.append("\n**Other relationships:**")
                for pair, corr in correlations.items():
                    if pair != strongest_pair[0]:
                        parts.append(f"• {pair}: {self.interpret_correlation_strength(corr)}")

        return "\n".join(parts)

    def generate_regression_summary(self, regression_result: Dict[str, Any]) -> str:
        """Generate plain language summary of regression model performance."""
        if not regression_result or 'r_squared' not in regression_result:
            raise ValueError("Invalid regression result")

        parts = []
        r_squared = regression_result['r_squared']

        if self.config.language_style == "technical":
            parts.append(f"{self._emoji('📈')}Regression Model Results:")
            parts.append(f"• R² = {r_squared:.3f}")
            parts.append(f"• Model explains {self.format_percentage(r_squared)} of variance")

            if 'coefficients' in regression_result:
                parts.append("\n**Coefficients:**")
                for var, coef in regression_result['coefficients'].items():
                    parts.append(f"• {var}: {coef:.4f}")
        else:
            parts.append(f"{self._emoji('📈')}Model Performance:")

            # Interpret R-squared
            if r_squared >= 0.9:
                quality = "excellent"
            elif r_squared >= 0.7:
                quality = "good"
            elif r_squared >= 0.5:
                quality = "moderate"
            else:
                quality = "weak"

            parts.append(f"• The model has {quality} predictive power")
            parts.append(f"• It explains {self.format_percentage(r_squared)} of the variation in the data")

            # Explain significant predictors
            if 'p_values' in regression_result and 'coefficients' in regression_result:
                significant = [var for var, p in regression_result['p_values'].items() if p < 0.05]
                if significant:
                    parts.append(f"\n**Key predictors:** {', '.join(significant)}")
                    for var in significant:
                        coef = regression_result['coefficients'].get(var)
                        if coef:
                            direction = "increases" if coef > 0 else "decreases"
                            parts.append(f"• Higher {var} {direction} the outcome")

        return "\n".join(parts)

    def generate_ml_training_summary(self, training_result: Dict[str, Any]) -> str:
        """Generate plain language summary of ML training results."""
        if not training_result or 'metrics' not in training_result:
            raise ValueError("Invalid training result")

        metrics = training_result['metrics']
        if not metrics:
            raise ValueError("Metrics cannot be empty")

        parts = []
        model_type = training_result.get('model_type', 'Unknown')

        if self.config.language_style == "technical":
            parts.append(f"{self._emoji('🤖')}Model Training Complete:")
            parts.append(f"• Type: {model_type}")
            parts.append(f"\n**Metrics:**")
            for metric, value in metrics.items():
                parts.append(f"• {metric}: {value:.4f}")

            if 'training_time' in training_result:
                parts.append(f"\n• Training time: {training_result['training_time']:.1f}s")
        else:
            parts.append(f"{self._emoji('🤖')}Your {model_type.replace('_', ' ')} model is ready!")

            # Interpret accuracy
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                acc_pct = self.format_percentage(acc)

                if acc >= 0.95:
                    quality = "excellent"
                elif acc >= 0.85:
                    quality = "very good"
                elif acc >= 0.75:
                    quality = "good"
                elif acc >= 0.65:
                    quality = "decent"
                else:
                    quality = "needs improvement"

                parts.append(f"\n**Performance:** {quality}")
                parts.append(f"• The model correctly predicts {acc_pct} of cases")

            # Additional metrics
            if 'precision' in metrics and 'recall' in metrics:
                prec = metrics['precision']
                rec = metrics['recall']

                if prec > rec + 0.1:
                    parts.append(f"• The model is more conservative (fewer false positives)")
                elif rec > prec + 0.1:
                    parts.append(f"• The model is more inclusive (catches more cases)")
                else:
                    parts.append(f"• The model is well-balanced")

            if 'training_time' in training_result:
                time_sec = training_result['training_time']
                if time_sec < 10:
                    parts.append(f"\n• Training was quick ({time_sec:.1f} seconds)")
                elif time_sec < 60:
                    parts.append(f"\n• Training took {time_sec:.1f} seconds")
                else:
                    parts.append(f"\n• Training took {time_sec/60:.1f} minutes")

        return "\n".join(parts)

    def generate_ml_prediction_summary(self, predictions: Dict[str, Any]) -> str:
        """Generate plain language summary of ML predictions."""
        if not predictions or 'n_predictions' not in predictions:
            raise ValueError("Invalid predictions")

        parts = []
        n_pred = predictions['n_predictions']

        if self.config.language_style == "technical":
            parts.append(f"{self._emoji('🔮')}Prediction Results:")
            parts.append(f"• Generated {n_pred} predictions")

            if 'prediction_summary' in predictions:
                summary = predictions['prediction_summary']
                parts.append(f"\n**Summary Statistics:**")
                for stat, value in summary.items():
                    parts.append(f"• {stat}: {value:.2f}")
        else:
            parts.append(f"{self._emoji('🔮')}Generated {self.format_large_number(n_pred)} predictions")

            if 'prediction_summary' in predictions:
                summary = predictions['prediction_summary']
                mean = summary.get('mean')
                median = summary.get('median')
                min_val = summary.get('min')
                max_val = summary.get('max')

                if mean is not None:
                    parts.append(f"\n• Average prediction: {mean:.2f}")

                if min_val is not None and max_val is not None:
                    parts.append(f"• Range: {min_val:.2f} to {max_val:.2f}")

                    range_size = max_val - min_val
                    if mean and range_size < 0.2 * mean:
                        parts.append(f"• Predictions are fairly consistent")
                    else:
                        parts.append(f"• Predictions vary significantly")

        return "\n".join(parts)

    def interpret_correlation_strength(self, correlation: float) -> str:
        """Interpret correlation coefficient strength in plain language."""
        abs_corr = abs(correlation)
        direction = "negative" if correlation < 0 else "positive"

        if abs_corr >= 0.8:
            strength = "very strong"
        elif abs_corr >= 0.6:
            strength = "strong"
        elif abs_corr >= 0.4:
            strength = "moderate"
        elif abs_corr >= 0.2:
            strength = "weak"
        else:
            strength = "very weak"

        if correlation < 0:
            return f"{strength} negative correlation (inverse relationship)"
        else:
            return f"{strength} positive correlation"

    def interpret_p_value(self, p_value: float) -> str:
        """Interpret p-value significance in plain language."""
        if p_value < 0.01:
            return "highly significant (very strong evidence)"
        elif p_value < 0.05:
            return "statistically significant"
        elif p_value < 0.1:
            return "marginally significant (borderline)"
        else:
            return "not statistically significant (no strong evidence)"

    def format_percentage(self, value: float) -> str:
        """Format decimal as percentage string."""
        pct = value * 100
        # Remove trailing .0 for whole numbers
        if pct == int(pct):
            return f"{int(pct)}%"
        return f"{pct:.1f}%"

    def format_large_number(self, number: int) -> str:
        """Format large numbers with thousand/million suffixes."""
        if number >= 1_000_000:
            return f"{number / 1_000_000:.1f} million"
        elif number >= 1_000:
            return f"{number / 1_000:.1f} thousand"
        else:
            return str(number)
