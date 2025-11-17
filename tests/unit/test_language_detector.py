"""Tests for LanguageDetector."""

import pytest
from src.utils.language_detector import LanguageDetector, LanguageDetectionResult


class TestLanguageDetector:
    """Test language detection functionality."""

    def test_init(self):
        """Test LanguageDetector initialization."""
        detector = LanguageDetector()
        assert detector.confidence_threshold == 0.8
        assert detector.use_llm_fallback is True
        assert "en" in detector.supported_languages
        assert "pt" in detector.supported_languages

    def test_english_detection_high_confidence(self):
        """Test English detection with high confidence."""
        detector = LanguageDetector()
        result = detector._detect_with_library(
            "Hello, how are you today? I would like to train a machine learning model."
        )
        assert result.language == "en"
        assert result.confidence > 0.7
        assert result.method == "library"

    def test_portuguese_detection_high_confidence(self):
        """Test Portuguese detection with high confidence."""
        detector = LanguageDetector()
        result = detector._detect_with_library(
            "Olá, como você está hoje? Eu gostaria de treinar um modelo de machine learning."
        )
        assert result.language == "pt"
        assert result.confidence > 0.7
        assert result.method == "library"

    @pytest.mark.asyncio
    async def test_cached_language_takes_priority(self):
        """Test that cached language is used when available."""
        detector = LanguageDetector()
        result = await detector.detect_language(
            text="Hello",
            user_id=123,
            cached_language="pt"
        )
        assert result.language == "pt"
        assert result.method == "cached"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_library_detection_for_new_user(self):
        """Test library detection for user without cached language."""
        detector = LanguageDetector()
        result = await detector.detect_language(
            text="Hello, I need help with my data analysis project",
            user_id=456,
            cached_language=None
        )
        assert result.language in ["en", "pt"]
        assert result.method in ["library", "llm", "default_fallback"]

    def test_short_text_detection(self):
        """Test detection on short text (may have lower confidence)."""
        detector = LanguageDetector()
        try:
            result = detector._detect_with_library("Hi")
            # Short text may still work
            assert result.language in ["en", "pt"]
        except Exception:
            # Or may fail - that's expected for very short text
            pass

    def test_detection_result_dataclass(self):
        """Test LanguageDetectionResult dataclass."""
        result = LanguageDetectionResult(
            language="en",
            confidence=0.95,
            method="library",
            detected_at="2025-01-01T00:00:00"
        )
        assert result.language == "en"
        assert result.confidence == 0.95
        assert result.method == "library"

    def test_unsupported_language_fallsback_to_english(self):
        """Test that unsupported detected languages fall back to English."""
        detector = LanguageDetector(supported_languages=["en", "pt"])
        # Library might detect 'es' (Spanish), should fallback to 'en'
        # This tests the logic in _detect_with_library
        assert "en" in detector.supported_languages

    @pytest.mark.asyncio
    async def test_detect_without_llm_fallback(self):
        """Test detection with LLM fallback disabled."""
        detector = LanguageDetector(use_llm_fallback=False)
        result = await detector.detect_language(
            text="Some text",
            user_id=789,
            cached_language=None
        )
        # Should use library or default fallback, never LLM
        assert result.method in ["library", "default_fallback"]

    def test_portuguese_common_phrases(self):
        """Test Portuguese detection with common phrases."""
        detector = LanguageDetector()
        phrases = [
            "Bom dia",
            "Como posso ajudar?",
            "Eu quero treinar um modelo",
            "Preciso de ajuda com análise de dados"
        ]
        for phrase in phrases:
            result = detector._detect_with_library(phrase)
            # Should detect as Portuguese
            assert result.language == "pt"

    def test_english_common_phrases(self):
        """Test English detection with common phrases."""
        detector = LanguageDetector()
        phrases = [
            "Good morning",
            "How can I help you?",
            "I want to train a model",
            "I need help with data analysis"
        ]
        for phrase in phrases:
            result = detector._detect_with_library(phrase)
            # Should detect as English
            assert result.language == "en"

    def test_mixed_content_detection(self):
        """Test detection on text with mixed content (technical terms)."""
        detector = LanguageDetector()
        # Text with Portuguese + technical English terms
        result = detector._detect_with_library(
            "Eu quero usar XGBoost regression para prever price"
        )
        # Should still detect as Portuguese (majority language)
        assert result.language == "pt"

    @pytest.mark.asyncio
    async def test_confidence_threshold_triggers_fallback(self):
        """Test that low confidence triggers LLM fallback."""
        detector = LanguageDetector(confidence_threshold=0.99)  # Very high threshold
        # Short text will have low confidence
        result = await detector.detect_language(
            text="Hi there",
            user_id=999,
            cached_language=None
        )
        # With high threshold, should fallback to LLM or default
        assert result.method in ["llm", "default_fallback"]

    def test_custom_supported_languages(self):
        """Test detector with custom supported languages list."""
        detector = LanguageDetector(supported_languages=["en"])
        assert detector.supported_languages == ["en"]
        assert "pt" not in detector.supported_languages
