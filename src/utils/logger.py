"""
Logging configuration for the Statistical Modeling Agent.

This module provides structured logging with proper formatting and levels
as specified in the CLAUDE.md guidelines.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "statistical_modeling_agent",
    level: str = "INFO",
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up structured logging with proper formatting.

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handlers if logger is already configured
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper()))

    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "statistical_modeling_agent") -> logging.Logger:
    """
    Get existing logger or create new one with default configuration.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)