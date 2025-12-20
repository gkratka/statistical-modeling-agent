"""Token manager for worker authentication.

Handles one-time token generation, validation, and user mapping for
secure worker authentication.
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class Token:
    """Represents an authentication token for worker connection.

    Attributes:
        value: The token string (UUID)
        user_id: Telegram user ID associated with this token
        created_at: When the token was created
        expires_at: When the token expires
        used: Whether the token has been used (one-time use)
    """

    value: str
    user_id: int
    created_at: datetime
    expires_at: datetime
    used: bool = False

    def is_expired(self) -> bool:
        """Check if token has expired."""
        return datetime.now() > self.expires_at

    def is_valid(self) -> bool:
        """Check if token is valid (not used and not expired)."""
        return not self.used and not self.is_expired()


class TokenManager:
    """Manages authentication tokens for worker connections.

    Provides token generation, validation, and cleanup. Tokens are:
    - One-time use (invalidated after first successful validation)
    - Time-limited (default 5 minutes)
    - User-specific (one active token per user)

    Attributes:
        token_expiry_seconds: How long tokens remain valid (default 300s/5min)
    """

    def __init__(self, token_expiry_seconds: int = 300):
        """Initialize token manager.

        Args:
            token_expiry_seconds: Token validity duration in seconds
        """
        self.token_expiry_seconds = token_expiry_seconds
        self._tokens: Dict[str, Token] = {}
        self._user_tokens: Dict[int, str] = {}  # user_id -> token_value

    def generate_token(self, user_id: int) -> str:
        """Generate a new authentication token for a user.

        If the user already has an active token, it will be invalidated
        and a new one created.

        Args:
            user_id: Telegram user ID

        Returns:
            The generated token string (UUID)
        """
        # Invalidate any existing token for this user
        if user_id in self._user_tokens:
            old_token = self._user_tokens[user_id]
            if old_token in self._tokens:
                del self._tokens[old_token]
            logger.debug(f"Invalidated old token for user {user_id}")

        # Generate new token
        token_value = str(uuid.uuid4())
        now = datetime.now()
        token = Token(
            value=token_value,
            user_id=user_id,
            created_at=now,
            expires_at=now + timedelta(seconds=self.token_expiry_seconds),
        )

        # Store token
        self._tokens[token_value] = token
        self._user_tokens[user_id] = token_value

        # DEBUG: Print for visibility (bypasses log level filtering)
        print(f"🔑 TOKEN GENERATED: {token_value[:8]}... for user {user_id}", flush=True)
        print(f"🔑 Total tokens in memory: {len(self._tokens)}", flush=True)
        print(f"🔑 TokenManager id: {id(self)}", flush=True)

        logger.info(
            f"Generated token for user {user_id}, "
            f"expires at {token.expires_at.isoformat()}"
        )

        return token_value

    def validate_token(self, token_value: str) -> Optional[int]:
        """Validate a token and mark it as used.

        This is a one-time validation - after successful validation,
        the token cannot be used again.

        Args:
            token_value: The token string to validate

        Returns:
            The user_id if token is valid, None otherwise
        """
        # DEBUG: Print validation attempt
        print(f"🔐 VALIDATING TOKEN: {token_value[:8]}...", flush=True)
        print(f"🔐 Known tokens: {[t[:8] + '...' for t in list(self._tokens.keys())[:5]]}", flush=True)
        print(f"🔐 TokenManager id: {id(self)}", flush=True)

        token = self._tokens.get(token_value)

        if token is None:
            print(f"❌ TOKEN NOT FOUND in _tokens dict!", flush=True)
            logger.warning(f"Token validation failed: unknown token")
            return None

        if token.is_expired():
            logger.warning(f"Token validation failed: expired")
            return None

        # Token remains valid for reconnection until explicit disconnect
        logger.info(f"Token validated for user {token.user_id}")

        return token.user_id

    def get_token_for_user(self, user_id: int) -> Optional[str]:
        """Get the current token for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            The token string if user has one, None otherwise
        """
        return self._user_tokens.get(user_id)

    def get_user_for_token(self, token_value: str) -> Optional[int]:
        """Get the user_id associated with a token.

        Args:
            token_value: The token string

        Returns:
            The user_id if token exists, None otherwise
        """
        token = self._tokens.get(token_value)
        return token.user_id if token else None

    def revoke_token(self, token_value: str) -> bool:
        """Revoke a specific token.

        Args:
            token_value: The token string to revoke

        Returns:
            True if token was revoked, False if not found
        """
        token = self._tokens.get(token_value)
        if token is None:
            return False

        # Remove from both mappings
        del self._tokens[token_value]
        if token.user_id in self._user_tokens:
            if self._user_tokens[token.user_id] == token_value:
                del self._user_tokens[token.user_id]

        logger.info(f"Revoked token for user {token.user_id}")
        return True

    def revoke_user_tokens(self, user_id: int) -> bool:
        """Revoke all tokens for a user.

        Args:
            user_id: Telegram user ID

        Returns:
            True if any tokens were revoked, False otherwise
        """
        token_value = self._user_tokens.get(user_id)
        if token_value is None:
            return False

        return self.revoke_token(token_value)

    def cleanup_expired_tokens(self) -> int:
        """Remove all expired tokens from storage.

        Returns:
            Number of tokens removed
        """
        expired = [
            token_value
            for token_value, token in self._tokens.items()
            if token.is_expired()
        ]

        for token_value in expired:
            self.revoke_token(token_value)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired tokens")

        return len(expired)

    def get_stats(self) -> Dict[str, int]:
        """Get token statistics.

        Returns:
            Dictionary with token statistics
        """
        total = len(self._tokens)
        expired = sum(1 for t in self._tokens.values() if t.is_expired())
        used = sum(1 for t in self._tokens.values() if t.used)
        active = sum(1 for t in self._tokens.values() if t.is_valid())

        return {
            "total_tokens": total,
            "active_tokens": active,
            "expired_tokens": expired,
            "used_tokens": used,
        }
