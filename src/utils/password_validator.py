"""Password validation for file path access control.

This module provides secure password validation with rate limiting, audit logging,
and session-based access control for non-whitelisted file paths.

Security Features:
    - Configurable password (default: 'senha123', overridable via env)
    - Rate limiting: Max 3 attempts per session
    - Exponential backoff: 2s, 5s, 10s delays
    - Audit logging: All attempts logged to separate auth.log
    - Session timeout: 5 minutes for password prompt
    - User isolation: Separate tracking per user_id

Usage:
    validator = PasswordValidator()
    is_valid, error_msg = validator.validate_password(
        user_id=12345,
        password_input="senha123",
        path="/path/to/file"
    )
"""

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Create auth logger for security events (separate from main logger)
auth_logger = logging.getLogger('auth')
auth_logger.setLevel(logging.INFO)

# Ensure logs directory exists (may fail on read-only filesystems like Railway)
try:
    log_dir = Path('data/logs')
    log_dir.mkdir(parents=True, exist_ok=True)

    # Add file handler for auth events
    auth_handler = logging.FileHandler(log_dir / 'auth.log')
    auth_handler.setFormatter(logging.Formatter(
        '%(asctime)s - [AUTH] - %(levelname)s - %(message)s'
    ))
    auth_logger.addHandler(auth_handler)
except (OSError, PermissionError) as e:
    # Fall back to stdout logging on read-only filesystems
    logger.warning(f"Could not create auth log file: {e}. Using stdout only.")


@dataclass
class PasswordAttempt:
    """Track password attempts for rate limiting.

    Attributes:
        user_id: User identifier
        attempt_count: Number of attempts made
        last_attempt: Timestamp of last attempt (seconds since epoch)
        locked_until: Timestamp when lockout expires (seconds since epoch)
        session_start: Timestamp when session started (seconds since epoch)
    """
    user_id: int
    attempt_count: int = 0
    last_attempt: Optional[float] = None
    locked_until: Optional[float] = None
    session_start: float = field(default_factory=time.time)


class PasswordValidator:
    """Password validation with rate limiting and audit logging.

    Security Features:
        - Configurable password (default: 'senha123', overridable via env)
        - Rate limiting: Max 3 attempts per session
        - Exponential backoff: 2s, 5s, 10s delays
        - Audit logging: All attempts logged
        - Session timeout: 5 minutes for password prompt
        - User isolation: Separate tracking per user_id

    Configuration:
        DEFAULT_PASSWORD: Default password ('senha123')
        MAX_ATTEMPTS: Maximum attempts before lockout (3)
        BACKOFF_DELAYS: Exponential backoff delays in seconds [2, 5, 10]
        SESSION_TIMEOUT: Password prompt expires after 5 minutes (300s)

    Example:
        >>> validator = PasswordValidator()
        >>> is_valid, error = validator.validate_password(
        ...     user_id=12345,
        ...     password_input="senha123",
        ...     path="/data/file.csv"
        ... )
        >>> if is_valid:
        ...     print("Access granted")
    """

    # Configuration constants
    DEFAULT_PASSWORD = "senha123"
    MAX_ATTEMPTS = 3
    BACKOFF_DELAYS = [2, 5, 10]  # seconds
    SESSION_TIMEOUT = 300  # 5 minutes

    def __init__(self, password: Optional[str] = None):
        """Initialize password validator.

        Args:
            password: Password to validate against. If None, uses environment
                     variable FILE_PATH_PASSWORD or default 'senha123'.
        """
        # Priority: 1) Explicit param, 2) Environment variable, 3) Default
        if password is None:
            password = os.getenv('FILE_PATH_PASSWORD', self.DEFAULT_PASSWORD)

        self.password = password
        self._attempts: Dict[int, PasswordAttempt] = {}

    def validate_password(
        self,
        user_id: int,
        password_input: str,
        path: str
    ) -> Tuple[bool, Optional[str]]:
        """Validate password with rate limiting.

        Args:
            user_id: User ID attempting access
            password_input: Password provided by user
            path: File path being accessed (for logging)

        Returns:
            Tuple of (is_valid, error_message):
                - (True, None) if password correct
                - (False, error_message) if incorrect or rate limited

        Security Notes:
            - All attempts are logged to auth.log
            - Rate limiting enforced with exponential backoff
            - Session timeout enforced (5 minutes)
            - Lockout after MAX_ATTEMPTS (60 second lockout)
        """
        # Get or create attempt record
        if user_id not in self._attempts:
            self._attempts[user_id] = PasswordAttempt(user_id=user_id)

        attempt = self._attempts[user_id]
        current_time = time.time()

        # Check if session expired
        if current_time - attempt.session_start > self.SESSION_TIMEOUT:
            auth_logger.warning(
                f"Session timeout: user={user_id}, "
                f"elapsed={current_time - attempt.session_start:.1f}s"
            )
            self._attempts[user_id] = PasswordAttempt(user_id=user_id)
            return False, (
                "Password prompt expired. Please try again with /train or /predict."
            )

        # Check if locked due to rate limit
        if attempt.locked_until and current_time < attempt.locked_until:
            wait_seconds = int(attempt.locked_until - current_time)
            auth_logger.warning(
                f"Rate limit active: user={user_id}, wait={wait_seconds}s"
            )
            return False, f"Too many failed attempts. Please wait {wait_seconds} seconds."

        # Check backoff delay
        if attempt.last_attempt:
            delay_index = min(attempt.attempt_count - 1, len(self.BACKOFF_DELAYS) - 1)
            required_delay = self.BACKOFF_DELAYS[delay_index]
            elapsed = current_time - attempt.last_attempt

            if elapsed < required_delay:
                wait_seconds = int(required_delay - elapsed)
                return False, f"Please wait {wait_seconds} seconds before trying again."

        # Increment attempt counter
        attempt.attempt_count += 1
        attempt.last_attempt = current_time

        # Validate password
        is_valid = password_input == self.password

        if is_valid:
            auth_logger.info(
                f"Access granted: user={user_id}, path={path}, "
                f"attempts={attempt.attempt_count}"
            )
            # Reset attempts on success
            del self._attempts[user_id]
            return True, None
        else:
            auth_logger.warning(
                f"Failed attempt: user={user_id}, path={path}, "
                f"attempts={attempt.attempt_count}/{self.MAX_ATTEMPTS}"
            )

            # Check if max attempts reached
            if attempt.attempt_count >= self.MAX_ATTEMPTS:
                attempt.locked_until = current_time + 60  # 60 second lockout
                auth_logger.error(
                    f"Rate limit exceeded: user={user_id}, locked_for=60s"
                )
                return False, (
                    f"Maximum attempts ({self.MAX_ATTEMPTS}) exceeded. "
                    f"Access locked for 60 seconds."
                )

            remaining = self.MAX_ATTEMPTS - attempt.attempt_count
            return False, (
                f"Incorrect password. {remaining} attempt(s) remaining."
            )

    def reset_attempts(self, user_id: int) -> None:
        """Reset attempt counter for user (e.g., on workflow restart).

        Args:
            user_id: User ID to reset attempts for
        """
        if user_id in self._attempts:
            auth_logger.info(f"Attempts reset: user={user_id}")
            del self._attempts[user_id]

    def get_attempt_count(self, user_id: int) -> int:
        """Get current attempt count for user.

        Args:
            user_id: User ID to check

        Returns:
            Number of attempts made (0 if no attempts)
        """
        if user_id not in self._attempts:
            return 0
        return self._attempts[user_id].attempt_count
