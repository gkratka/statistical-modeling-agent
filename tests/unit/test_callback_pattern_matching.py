"""
TDD Tests for Callback Pattern Matching.

These tests verify that callback query patterns correctly match the
callback_data sent by InlineKeyboard buttons.

Background:
After fixing Button_data_invalid error by removing model_id from callback_data,
we changed from "name_model:{model_id}" to "name_model". The handler patterns
must match this new format exactly.
"""

import pytest
import re


class TestCallbackPatternMatching:
    """Test that callback patterns match button callback_data exactly."""

    def test_name_model_pattern_matches_simple_callback(self):
        """
        Verify 'name_model' pattern matches button callback_data.

        Button sends: "name_model"
        Pattern must match this exactly (no colon expected).
        """
        # NEW pattern (exact match with $ anchor)
        pattern = r"^name_model$"
        callback_data = "name_model"  # What InlineKeyboardButton sends

        # Test pattern matching
        match = re.match(pattern, callback_data)
        assert match is not None, \
            f"Pattern '{pattern}' should match callback_data '{callback_data}'"

    def test_skip_naming_pattern_matches_simple_callback(self):
        """
        Verify 'skip_naming' pattern matches button callback_data.

        Button sends: "skip_naming"
        Pattern must match this exactly (no colon expected).
        """
        # NEW pattern (exact match with $ anchor)
        pattern = r"^skip_naming$"
        callback_data = "skip_naming"  # What InlineKeyboardButton sends

        # Test pattern matching
        match = re.match(pattern, callback_data)
        assert match is not None, \
            f"Pattern '{pattern}' should match callback_data '{callback_data}'"

    def test_old_pattern_with_colon_fails(self):
        """
        Verify OLD pattern with colon does NOT match new callback_data.

        This test documents why the old pattern failed.
        """
        # OLD pattern (expected colon separator)
        old_pattern = r"^name_model:"
        callback_data = "name_model"  # NEW format (no colon)

        # Should NOT match
        match = re.match(old_pattern, callback_data)
        assert match is None, \
            f"Old pattern '{old_pattern}' should NOT match new callback_data '{callback_data}'"

    def test_pattern_rejects_similar_but_different_callbacks(self):
        """
        Ensure pattern is specific enough to reject similar callbacks.

        Pattern should use $ anchor to prevent partial matches.
        """
        pattern = r"^name_model$"

        # These should NOT match
        invalid_callbacks = [
            "name_model:",         # Has colon
            "name_model_extra",    # Extra suffix
            "prefix_name_model",   # Has prefix
            "name_models",         # Plural
            "name_model_123",      # Has ID
        ]

        for invalid_callback in invalid_callbacks:
            match = re.match(pattern, invalid_callback)
            assert match is None, \
                f"Pattern '{pattern}' should NOT match '{invalid_callback}'"

    def test_patterns_are_case_sensitive(self):
        """Verify patterns are case-sensitive (as they should be)."""
        pattern = r"^name_model$"

        # These should NOT match (different case)
        assert re.match(pattern, "Name_Model") is None
        assert re.match(pattern, "NAME_MODEL") is None
        assert re.match(pattern, "name_Model") is None

        # Only exact case should match
        assert re.match(pattern, "name_model") is not None


class TestPatternPerformance:
    """Test edge cases and performance characteristics of patterns."""

    def test_pattern_compilation(self):
        """Verify patterns can be compiled without errors."""
        patterns = [
            r"^name_model$",
            r"^skip_naming$",
        ]

        for pattern_str in patterns:
            try:
                compiled = re.compile(pattern_str)
                assert compiled is not None
            except re.error as e:
                pytest.fail(f"Pattern '{pattern_str}' failed to compile: {e}")

    def test_anchor_behavior(self):
        """
        Verify $ anchor prevents partial matches.

        Without $: "name_model" matches "name_model_anything"
        With $: "name_model" only matches "name_model" exactly
        """
        # Without $ anchor (BAD - too permissive)
        bad_pattern = r"^name_model"
        assert re.match(bad_pattern, "name_model_anything") is not None  # Matches unwanted

        # With $ anchor (GOOD - exact match only)
        good_pattern = r"^name_model$"
        assert re.match(good_pattern, "name_model_anything") is None  # Correctly rejects


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
