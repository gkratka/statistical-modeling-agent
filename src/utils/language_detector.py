"""Language detection service with hybrid LLM fallback."""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

try:
    from langdetect import detect_langs, LangDetectException
except ImportError:
    # Fallback if langdetect not installed yet
    detect_langs = None
    LangDetectException = Exception

from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class LanguageDetectionResult:
    """Result of language detection."""
    language: str  # ISO 639-1 code (en, pt)
    confidence: float  # 0-1
    method: str  # "library", "llm", "cached"
    detected_at: str  # ISO timestamp


class LanguageDetector:
    """Hybrid language detector with library + LLM fallback."""

    def __init__(
        self,
        confidence_threshold: float = 0.8,
        use_llm_fallback: bool = True,
        supported_languages: Optional[list[str]] = None
    ):
        """
        Initialize language detector.

        Args:
            confidence_threshold: Minimum confidence for library detection
            use_llm_fallback: Whether to use LLM for low-confidence cases
            supported_languages: List of supported language codes
        """
        self.confidence_threshold = confidence_threshold
        self.use_llm_fallback = use_llm_fallback
        self.supported_languages = supported_languages or ["en", "pt"]

    async def detect_language(
        self,
        text: str,
        user_id: int,
        cached_language: Optional[str] = None
    ) -> LanguageDetectionResult:
        """
        Detect language from text with hybrid approach.

        Args:
            text: Message text to analyze
            user_id: User ID for LLM API tracking
            cached_language: Previously detected language (if any)

        Returns:
            LanguageDetectionResult with detected language and confidence
        """
        # Strategy 1: Use cached language if available (99% of cases)
        if cached_language and cached_language in self.supported_languages:
            return LanguageDetectionResult(
                language=cached_language,
                confidence=1.0,
                method="cached",
                detected_at=datetime.now().isoformat()
            )

        # Strategy 2: Library detection (langdetect)
        try:
            library_result = self._detect_with_library(text)

            if library_result.confidence >= self.confidence_threshold:
                return library_result

            logger.info(
                f"Low confidence library detection: {library_result.confidence:.2f}, "
                f"text_length={len(text)}"
            )

        except Exception as e:
            logger.warning(f"Library detection failed: {e}")

        # Strategy 3: LLM fallback (for ambiguous cases)
        if self.use_llm_fallback:
            return await self._detect_with_llm(text, user_id)

        # Fallback to default
        return LanguageDetectionResult(
            language="en",
            confidence=0.5,
            method="default_fallback",
            detected_at=datetime.now().isoformat()
        )

    def _detect_with_library(self, text: str) -> LanguageDetectionResult:
        """
        Detect language using langdetect library.

        Args:
            text: Text to analyze

        Returns:
            LanguageDetectionResult with library detection

        Raises:
            LangDetectException: If detection fails
        """
        if detect_langs is None:
            raise ImportError("langdetect not installed")

        # Quick Portuguese word detection (fallback for short text)
        portuguese_indicators = ['olá', 'você', 'preciso', 'ajuda', 'obrigado', 'por favor']
        text_lower = text.lower()
        if any(word in text_lower for word in portuguese_indicators):
            return LanguageDetectionResult(
                language="pt",
                confidence=0.9,
                method="keyword_match",
                detected_at=datetime.now().isoformat()
            )

        # langdetect works best with 20+ characters
        langs = detect_langs(text)

        if not langs:
            raise LangDetectException("No language detected")

        top_lang = langs[0]

        # Map langdetect codes to supported languages
        lang_code = top_lang.lang
        if lang_code not in self.supported_languages:
            # Default to English if detected language not supported
            lang_code = "en"

        return LanguageDetectionResult(
            language=lang_code,
            confidence=top_lang.prob,
            method="library",
            detected_at=datetime.now().isoformat()
        )

    async def _detect_with_llm(
        self,
        text: str,
        user_id: int
    ) -> LanguageDetectionResult:
        """
        Fallback to LLM for ambiguous cases.

        Args:
            text: Text to analyze
            user_id: User ID for tracking

        Returns:
            LanguageDetectionResult with LLM detection
        """
        try:
            from anthropic import AsyncAnthropic
            import os

            client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

            prompt = f"""Detect the language of this message. Respond with ONLY the ISO 639-1 code (en or pt).

Message: "{text}"

Language code:"""

            message = await client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )

            detected = message.content[0].text.strip().lower()

            # Validate response
            if detected not in self.supported_languages:
                logger.warning(f"LLM returned unsupported language: {detected}")
                detected = "en"

            return LanguageDetectionResult(
                language=detected,
                confidence=0.95,  # High confidence for LLM
                method="llm",
                detected_at=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"LLM detection failed: {e}")
            return LanguageDetectionResult(
                language="en",
                confidence=0.5,
                method="llm_fallback",
                detected_at=datetime.now().isoformat()
            )
