"""
XGBoost Availability Checker.

Utility to check if XGBoost can be loaded successfully, including runtime
dependencies like OpenMP (libomp).
"""

from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def check_xgboost_available() -> Tuple[bool, str]:
    """
    Check if XGBoost can be loaded and used.

    Returns:
        Tuple of (is_available, message):
        - is_available: True if XGBoost works, False otherwise
        - message: Success message with version or error description

    Examples:
        >>> available, msg = check_xgboost_available()
        >>> if available:
        ...     print(f"XGBoost ready: {msg}")
        ... else:
        ...     print(f"XGBoost unavailable: {msg}")
    """
    try:
        # Try importing xgboost
        import xgboost

        # Try importing the sklearn-compatible classes
        from xgboost import XGBClassifier, XGBRegressor

        # Try instantiating a model (this will fail if libomp is missing)
        _ = XGBClassifier(n_estimators=1)

        # If we got here, XGBoost is fully functional
        return (True, f"XGBoost {xgboost.__version__}")

    except Exception as e:
        error_str = str(e)

        # Detect specific error types
        if 'libomp' in error_str or 'OpenMP' in error_str.lower():
            return (False, "OpenMP runtime missing (run: brew install libomp)")

        elif 'XGBoost' in error_str and 'Library' in error_str:
            return (False, "XGBoost library load failed (system dependency issue)")

        elif 'ModuleNotFoundError' in error_str or 'No module named' in error_str:
            return (False, "XGBoost not installed (run: pip install xgboost>=1.7.0)")

        else:
            # Generic error
            return (False, f"Initialization failed: {error_str[:100]}")


def get_xgboost_setup_instructions() -> str:
    """
    Get platform-specific setup instructions for XGBoost.

    Returns:
        Formatted setup instructions string
    """
    import platform

    os_name = platform.system()

    if os_name == "Darwin":  # macOS
        return (
            "macOS Setup Instructions:\n\n"
            "1. Install Homebrew (if not already installed):\n"
            "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"\n\n"
            "2. Install OpenMP runtime:\n"
            "   brew install libomp\n\n"
            "3. Restart bot:\n"
            "   pkill -9 -f telegram_bot && ./scripts/dev_start.sh\n\n"
            "Full guide: XGBOOST_SETUP.md"
        )

    elif os_name == "Linux":
        return (
            "Linux Setup Instructions:\n\n"
            "1. Install OpenMP runtime:\n"
            "   Ubuntu/Debian: sudo apt-get install libgomp1\n"
            "   CentOS/RHEL: sudo yum install libgomp\n\n"
            "2. Restart bot\n\n"
            "Full guide: XGBOOST_SETUP.md"
        )

    elif os_name == "Windows":
        return (
            "Windows Setup Instructions:\n\n"
            "1. XGBoost should work out-of-the-box on Windows\n"
            "2. If issues persist, try reinstalling:\n"
            "   pip uninstall xgboost\n"
            "   pip install xgboost>=1.7.0\n\n"
            "Full guide: XGBOOST_SETUP.md"
        )

    else:
        return (
            "Unknown Platform Setup:\n\n"
            "Please refer to XGBoost documentation:\n"
            "https://xgboost.readthedocs.io/en/latest/install.html\n\n"
            "Full guide: XGBOOST_SETUP.md"
        )


def log_xgboost_status() -> None:
    """
    Log XGBoost availability status to logger.

    This is useful for bot startup diagnostics.
    """
    available, message = check_xgboost_available()

    if available:
        logger.info(f"✓ XGBoost available: {message}")
    else:
        logger.warning(f"⚠️  XGBoost unavailable: {message}")
        logger.warning(f"    Users selecting XGBoost models will see setup instructions")
        logger.info(f"    Alternative: sklearn Gradient Boosting works without dependencies")
