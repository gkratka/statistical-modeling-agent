"""
ML Engine Configuration Management.

This module provides configuration loading and validation for the ML Engine,
including model defaults, hyperparameter ranges, and resource limits.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import yaml

from src.utils.exceptions import ConfigurationError


@dataclass
class MLEngineConfig:
    """Configuration for ML Engine."""

    # Model storage
    models_dir: Path
    max_models_per_user: int
    max_model_size_mb: int

    # Training limits
    max_training_time: int  # seconds
    max_memory_mb: int
    min_training_samples: int

    # Data preprocessing defaults
    default_test_size: float
    default_cv_folds: int
    default_missing_strategy: str
    default_scaling: str

    # Model hyperparameters
    default_hyperparameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    hyperparameter_ranges: Dict[str, list] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate configuration values."""
        # Validate numeric ranges
        if self.max_training_time <= 0:
            raise ConfigurationError(
                "max_training_time must be positive",
                "ml_engine.max_training_time"
            )

        if self.min_training_samples < 2:
            raise ConfigurationError(
                "min_training_samples must be at least 2",
                "ml_engine.min_training_samples"
            )

        if not (0.0 < self.default_test_size < 1.0):
            raise ConfigurationError(
                "default_test_size must be between 0 and 1",
                "ml_engine.default_test_size"
            )

        if self.default_cv_folds < 2:
            raise ConfigurationError(
                "default_cv_folds must be at least 2",
                "ml_engine.default_cv_folds"
            )

        # Validate strategy options
        valid_missing_strategies = ["mean", "median", "drop"]
        if self.default_missing_strategy not in valid_missing_strategies:
            raise ConfigurationError(
                f"default_missing_strategy must be one of {valid_missing_strategies}",
                "ml_engine.default_missing_strategy"
            )

        valid_scaling_methods = ["standard", "minmax", "robust", "none"]
        if self.default_scaling not in valid_scaling_methods:
            raise ConfigurationError(
                f"default_scaling must be one of {valid_scaling_methods}",
                "ml_engine.default_scaling"
            )

        # Ensure models directory can be created
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ConfigurationError(
                f"Cannot create models directory: {e}",
                "ml_engine.models_dir"
            )

    @classmethod
    def from_yaml(cls, config_path: Path) -> "MLEngineConfig":
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            MLEngineConfig instance

        Raises:
            ConfigurationError: If config file invalid or missing
        """
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                "config_path"
            )
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Invalid YAML in configuration file: {e}",
                "config_path"
            )

        if "ml_engine" not in config:
            raise ConfigurationError(
                "Missing 'ml_engine' section in configuration",
                "ml_engine"
            )

        ml_config = config["ml_engine"]

        return cls(
            models_dir=Path(ml_config["models_dir"]),
            max_models_per_user=ml_config["max_models_per_user"],
            max_model_size_mb=ml_config["max_model_size_mb"],
            max_training_time=ml_config["max_training_time_seconds"],
            max_memory_mb=ml_config["max_memory_mb"],
            min_training_samples=ml_config["min_training_samples"],
            default_test_size=ml_config["default_test_size"],
            default_cv_folds=ml_config["default_cv_folds"],
            default_missing_strategy=ml_config["default_missing_strategy"],
            default_scaling=ml_config["default_scaling"],
            default_hyperparameters=ml_config.get("default_hyperparameters", {}),
            hyperparameter_ranges=ml_config.get("hyperparameter_ranges", {})
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "MLEngineConfig":
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            MLEngineConfig instance
        """
        return cls(
            models_dir=Path(config_dict.get("models_dir", "models")),
            max_models_per_user=config_dict.get("max_models_per_user", 50),
            max_model_size_mb=config_dict.get("max_model_size_mb", 100),
            max_training_time=config_dict.get("max_training_time_seconds", 300),
            max_memory_mb=config_dict.get("max_memory_mb", 2048),
            min_training_samples=config_dict.get("min_training_samples", 10),
            default_test_size=config_dict.get("default_test_size", 0.2),
            default_cv_folds=config_dict.get("default_cv_folds", 5),
            default_missing_strategy=config_dict.get("default_missing_strategy", "mean"),
            default_scaling=config_dict.get("default_scaling", "standard"),
            default_hyperparameters=config_dict.get("default_hyperparameters", {}),
            hyperparameter_ranges=config_dict.get("hyperparameter_ranges", {})
        )

    def get_default_hyperparameters(self, model_type: str) -> Dict[str, Any]:
        """
        Get default hyperparameters for a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary of default hyperparameters
        """
        return self.default_hyperparameters.get(model_type, {}).copy()

    def get_hyperparameter_range(self, parameter_name: str) -> Optional[Tuple]:
        """
        Get allowed range for a hyperparameter.

        Args:
            parameter_name: Name of hyperparameter

        Returns:
            Tuple of (min, max) or None if no range defined
        """
        range_list = self.hyperparameter_ranges.get(parameter_name)
        if range_list and len(range_list) == 2:
            return tuple(range_list)
        return None

    def to_dict(self) -> dict:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dictionary
        """
        return {
            "models_dir": str(self.models_dir),
            "max_models_per_user": self.max_models_per_user,
            "max_model_size_mb": self.max_model_size_mb,
            "max_training_time_seconds": self.max_training_time,
            "max_memory_mb": self.max_memory_mb,
            "min_training_samples": self.min_training_samples,
            "default_test_size": self.default_test_size,
            "default_cv_folds": self.default_cv_folds,
            "default_missing_strategy": self.default_missing_strategy,
            "default_scaling": self.default_scaling,
            "default_hyperparameters": self.default_hyperparameters,
            "hyperparameter_ranges": self.hyperparameter_ranges
        }

    @classmethod
    def get_default(cls) -> "MLEngineConfig":
        """
        Get default ML Engine configuration.

        Returns:
            MLEngineConfig instance with sensible defaults
        """
        return cls(
            models_dir=Path("models"),
            max_models_per_user=50,
            max_model_size_mb=100,
            max_training_time=300,  # 5 minutes
            max_memory_mb=2048,  # 2GB
            min_training_samples=10,
            default_test_size=0.2,
            default_cv_folds=5,
            default_missing_strategy="mean",
            default_scaling="standard",
            default_hyperparameters={
                "ridge": {"alpha": 1.0},
                "lasso": {"alpha": 1.0},
                "random_forest": {"n_estimators": 100},
                "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1},
                "mlp_regression": {"hidden_layers": [100], "max_iter": 200},
                "mlp_classification": {"hidden_layers": [100], "max_iter": 200}
            },
            hyperparameter_ranges={
                "alpha": [0.0001, 10.0],
                "n_estimators": [10, 500],
                "learning_rate": [0.001, 0.5],
                "max_depth": [1, 50]
            }
        )
