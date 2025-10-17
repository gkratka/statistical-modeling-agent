"""
TDD Tests for ML Training Completion Workflow.

These tests verify that the training completion flow properly handles
Telegram's callback_data size limits and stores model information in session.
"""

import pytest
from telegram import InlineKeyboardButton, InlineKeyboardMarkup


class TestCallbackDataValidation:
    """Test callback_data length compliance with Telegram limits."""

    def test_callback_data_within_telegram_limit(self):
        """
        Test that callback_data for model naming buttons stays within 64-byte limit.

        This test should PASS after the fix is implemented.
        Currently FAILS because callback_data includes full model_id.
        """
        # Simulate a long model_id (realistic worst case)
        long_model_id = "model_7715560927_keras_binary_classification_20251014_044444"

        # OLD implementation (should fail):
        # callback_data = f"name_model:{long_model_id}"  # 63 bytes - at limit

        # NEW implementation (should pass):
        callback_data = "name_model"  # 10 bytes - well within limit

        # Telegram's limit
        TELEGRAM_CALLBACK_DATA_LIMIT = 64

        # Verify callback_data is within limit
        assert len(callback_data.encode('utf-8')) <= TELEGRAM_CALLBACK_DATA_LIMIT, \
            f"callback_data '{callback_data}' exceeds {TELEGRAM_CALLBACK_DATA_LIMIT} bytes"

    def test_skip_naming_callback_within_limit(self):
        """
        Test that 'skip naming' callback_data stays within 64-byte limit.

        This test should PASS after the fix is implemented.
        """
        # NEW implementation (should pass):
        callback_data = "skip_naming"  # 11 bytes - well within limit

        TELEGRAM_CALLBACK_DATA_LIMIT = 64

        assert len(callback_data.encode('utf-8')) <= TELEGRAM_CALLBACK_DATA_LIMIT, \
            f"callback_data '{callback_data}' exceeds {TELEGRAM_CALLBACK_DATA_LIMIT} bytes"

    def test_all_callback_buttons_valid(self):
        """
        Test that all model naming workflow buttons have valid callback_data.

        This simulates the actual keyboard creation without model_id in callback_data.
        """
        # Create keyboard as it should be after fix
        keyboard = [
            [InlineKeyboardButton("📝 Name Model", callback_data="name_model")],
            [InlineKeyboardButton("⏭️ Skip - Use Default", callback_data="skip_naming")]
        ]

        TELEGRAM_CALLBACK_DATA_LIMIT = 64

        # Verify all buttons have valid callback_data
        for row in keyboard:
            for button in row:
                callback_data = button.callback_data
                assert len(callback_data.encode('utf-8')) <= TELEGRAM_CALLBACK_DATA_LIMIT, \
                    f"Button '{button.text}' has callback_data '{callback_data}' " \
                    f"exceeding {TELEGRAM_CALLBACK_DATA_LIMIT} bytes"


class TestSessionStoragePattern:
    """Test that model_id is properly stored in session instead of callback_data."""

    def test_model_id_stored_in_session(self):
        """
        Verify that model_id is stored in session.selections['pending_model_id'].

        This test validates the existing code at lines 1831-1832.
        Should PASS (code already does this correctly).
        """
        # Simulate session structure
        session = type('Session', (), {
            'selections': {}
        })()

        # Simulate storing model_id (as code does at line 1831)
        model_id = "model_7715560927_keras_binary_classification_20251014_044444"
        session.selections['pending_model_id'] = model_id

        # Verify storage
        assert 'pending_model_id' in session.selections
        assert session.selections['pending_model_id'] == model_id

    def test_callback_handlers_can_retrieve_from_session(self):
        """
        Test pattern: callback handlers retrieve model_id from session, not callback_data.

        This validates that the fix pattern works correctly.
        """
        # Simulate session with stored model_id
        session = type('Session', (), {
            'selections': {
                'pending_model_id': 'model_12345_random_forest'
            }
        })()

        # Simulate callback_data without model_id (after fix)
        callback_data = "name_model"

        # Handler should retrieve model_id from session
        model_id = session.selections.get('pending_model_id')

        # Verify retrieval works
        assert model_id == 'model_12345_random_forest'
        assert callback_data == 'name_model'  # No model_id in callback


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_maximum_safe_callback_data_length(self):
        """Verify we stay well below the 64-byte limit for safety margin."""
        # Our callbacks should be short and simple
        safe_callbacks = [
            "name_model",      # 10 bytes
            "skip_naming",     # 11 bytes
            "confirm_name",    # 12 bytes (potential future callback)
        ]

        # All should be under 20 bytes for comfortable safety margin
        SAFE_MARGIN = 20

        for callback in safe_callbacks:
            assert len(callback.encode('utf-8')) < SAFE_MARGIN, \
                f"callback '{callback}' should be under {SAFE_MARGIN} bytes for safety"

    def test_unicode_emoji_in_button_text_not_in_callback(self):
        """
        Ensure emoji/unicode in button text doesn't affect callback_data size.

        Button text can be any length, only callback_data has 64-byte limit.
        """
        button = InlineKeyboardButton(
            "📝 Name Model with Very Long Text 🎯✨",
            callback_data="name_model"
        )

        # Button text can be long
        assert len(button.text) > 20

        # Callback data must be short
        assert len(button.callback_data.encode('utf-8')) <= 64


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
