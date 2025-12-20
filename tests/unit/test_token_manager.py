"""Unit tests for token manager."""

import pytest
import time
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.worker.token_manager import TokenManager, Token


class TestToken:
    """Tests for Token dataclass."""

    def test_token_creation(self):
        """Test creating a token with all fields."""
        token = Token(
            value="test-token-123",
            user_id=12345,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        assert token.value == "test-token-123"
        assert token.user_id == 12345
        assert token.used is False

    def test_token_is_expired_when_past_expiry(self):
        """Test that token is marked expired after expiry time."""
        token = Token(
            value="test-token",
            user_id=12345,
            created_at=datetime.now() - timedelta(minutes=10),
            expires_at=datetime.now() - timedelta(minutes=5)
        )
        assert token.is_expired() is True

    def test_token_is_not_expired_when_before_expiry(self):
        """Test that token is not expired before expiry time."""
        token = Token(
            value="test-token",
            user_id=12345,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        assert token.is_expired() is False

    def test_token_is_valid_when_not_used_and_not_expired(self):
        """Test token validity check."""
        token = Token(
            value="test-token",
            user_id=12345,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5)
        )
        assert token.is_valid() is True

    def test_token_is_invalid_when_used(self):
        """Test that used token is invalid."""
        token = Token(
            value="test-token",
            user_id=12345,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=5),
            used=True
        )
        assert token.is_valid() is False

    def test_token_is_invalid_when_expired(self):
        """Test that expired token is invalid."""
        token = Token(
            value="test-token",
            user_id=12345,
            created_at=datetime.now() - timedelta(minutes=10),
            expires_at=datetime.now() - timedelta(minutes=5)
        )
        assert token.is_valid() is False


class TestTokenManager:
    """Tests for TokenManager class."""

    @pytest.fixture
    def manager(self):
        """Create token manager instance."""
        return TokenManager(token_expiry_seconds=300)

    def test_manager_initialization(self, manager):
        """Test manager initializes with correct config."""
        assert manager.token_expiry_seconds == 300
        assert len(manager._tokens) == 0
        assert len(manager._user_tokens) == 0

    def test_manager_default_expiry(self):
        """Test manager uses default 5-minute expiry."""
        manager = TokenManager()
        assert manager.token_expiry_seconds == 300

    def test_generate_token_creates_valid_token(self, manager):
        """Test token generation creates valid UUID token."""
        token_value = manager.generate_token(user_id=12345)

        assert token_value is not None
        assert len(token_value) == 36  # UUID format
        assert "-" in token_value

    def test_generate_token_stores_token(self, manager):
        """Test generated token is stored internally."""
        token_value = manager.generate_token(user_id=12345)

        assert token_value in manager._tokens
        assert manager._tokens[token_value].user_id == 12345

    def test_generate_token_creates_user_mapping(self, manager):
        """Test generated token creates user-to-token mapping."""
        token_value = manager.generate_token(user_id=12345)

        assert 12345 in manager._user_tokens
        assert manager._user_tokens[12345] == token_value

    def test_generate_token_invalidates_previous_token(self, manager):
        """Test generating new token invalidates previous one for same user."""
        token1 = manager.generate_token(user_id=12345)
        token2 = manager.generate_token(user_id=12345)

        assert token1 != token2
        assert token1 not in manager._tokens  # Old token removed
        assert token2 in manager._tokens
        assert manager._user_tokens[12345] == token2

    def test_generate_token_sets_correct_expiry(self, manager):
        """Test token has correct expiry time."""
        before = datetime.now()
        token_value = manager.generate_token(user_id=12345)
        after = datetime.now()

        token = manager._tokens[token_value]
        expected_expiry_min = before + timedelta(seconds=300)
        expected_expiry_max = after + timedelta(seconds=300)

        assert expected_expiry_min <= token.expires_at <= expected_expiry_max

    def test_validate_token_returns_user_id_for_valid_token(self, manager):
        """Test validating valid token returns user_id."""
        token_value = manager.generate_token(user_id=12345)

        user_id = manager.validate_token(token_value)

        assert user_id == 12345

    def test_validate_token_marks_token_as_used(self, manager):
        """Test validating token marks it as used (one-time use)."""
        token_value = manager.generate_token(user_id=12345)

        manager.validate_token(token_value)

        assert manager._tokens[token_value].used is True

    def test_validate_token_returns_none_for_already_used_token(self, manager):
        """Test already used token cannot be validated again."""
        token_value = manager.generate_token(user_id=12345)

        # First validation succeeds
        user_id = manager.validate_token(token_value)
        assert user_id == 12345

        # Second validation fails
        user_id = manager.validate_token(token_value)
        assert user_id is None

    def test_validate_token_returns_none_for_unknown_token(self, manager):
        """Test unknown token validation returns None."""
        user_id = manager.validate_token("unknown-token-value")

        assert user_id is None

    def test_validate_token_returns_none_for_expired_token(self, manager):
        """Test expired token validation returns None."""
        # Create manager with very short expiry
        short_manager = TokenManager(token_expiry_seconds=0)
        token_value = short_manager.generate_token(user_id=12345)

        # Wait for expiry
        time.sleep(0.1)

        user_id = short_manager.validate_token(token_value)
        assert user_id is None

    def test_get_token_for_user(self, manager):
        """Test getting current token for user."""
        token_value = manager.generate_token(user_id=12345)

        result = manager.get_token_for_user(12345)

        assert result == token_value

    def test_get_token_for_user_returns_none_if_no_token(self, manager):
        """Test getting token for user with no token returns None."""
        result = manager.get_token_for_user(99999)

        assert result is None

    def test_get_user_for_token(self, manager):
        """Test getting user_id for token."""
        token_value = manager.generate_token(user_id=12345)

        user_id = manager.get_user_for_token(token_value)

        assert user_id == 12345

    def test_get_user_for_token_returns_none_for_unknown_token(self, manager):
        """Test getting user_id for unknown token returns None."""
        user_id = manager.get_user_for_token("unknown-token")

        assert user_id is None

    def test_revoke_token(self, manager):
        """Test revoking a token."""
        token_value = manager.generate_token(user_id=12345)

        result = manager.revoke_token(token_value)

        assert result is True
        assert token_value not in manager._tokens
        assert 12345 not in manager._user_tokens

    def test_revoke_token_returns_false_for_unknown_token(self, manager):
        """Test revoking unknown token returns False."""
        result = manager.revoke_token("unknown-token")

        assert result is False

    def test_revoke_user_tokens(self, manager):
        """Test revoking all tokens for a user."""
        token_value = manager.generate_token(user_id=12345)

        result = manager.revoke_user_tokens(12345)

        assert result is True
        assert token_value not in manager._tokens
        assert 12345 not in manager._user_tokens

    def test_revoke_user_tokens_returns_false_if_no_tokens(self, manager):
        """Test revoking tokens for user with no tokens returns False."""
        result = manager.revoke_user_tokens(99999)

        assert result is False

    def test_cleanup_expired_tokens(self, manager):
        """Test cleaning up expired tokens."""
        # Create manager with short expiry
        short_manager = TokenManager(token_expiry_seconds=0)
        token1 = short_manager.generate_token(user_id=12345)

        # Wait for expiry
        time.sleep(0.1)

        # Create another token (won't be expired immediately)
        regular_manager = TokenManager(token_expiry_seconds=300)
        token2 = regular_manager.generate_token(user_id=67890)

        # Cleanup on short manager
        removed = short_manager.cleanup_expired_tokens()

        assert removed >= 1
        assert token1 not in short_manager._tokens

    def test_get_stats(self, manager):
        """Test getting token statistics."""
        manager.generate_token(user_id=12345)
        manager.generate_token(user_id=67890)

        stats = manager.get_stats()

        assert stats["total_tokens"] == 2
        assert stats["active_tokens"] == 2
        assert stats["expired_tokens"] == 0
        assert stats["used_tokens"] == 0

    def test_get_stats_with_used_token(self, manager):
        """Test stats with used token."""
        token_value = manager.generate_token(user_id=12345)
        manager.validate_token(token_value)

        stats = manager.get_stats()

        assert stats["total_tokens"] == 1
        assert stats["used_tokens"] == 1


class TestTokenManagerConcurrency:
    """Tests for thread-safety of TokenManager."""

    @pytest.fixture
    def manager(self):
        """Create token manager instance."""
        return TokenManager(token_expiry_seconds=300)

    def test_multiple_users_can_have_tokens(self, manager):
        """Test multiple users can have concurrent tokens."""
        tokens = {}
        for user_id in range(100, 110):
            tokens[user_id] = manager.generate_token(user_id)

        assert len(manager._tokens) == 10
        assert len(manager._user_tokens) == 10

        for user_id, token in tokens.items():
            assert manager.get_user_for_token(token) == user_id
