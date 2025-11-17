"""i18n manager for translation loading and message formatting."""

from pathlib import Path
from typing import Any, Optional

try:
    import i18n
except ImportError:
    # Fallback if python-i18n not installed yet
    i18n = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


class I18nManager:
    """Centralized i18n configuration and message retrieval."""

    _initialized = False

    @classmethod
    def initialize(cls, locales_dir: str, default_locale: str = "en"):
        """
        Initialize i18n library with configuration.

        Args:
            locales_dir: Path to locales directory
            default_locale: Default language code
        """
        if cls._initialized:
            return

        if i18n is None:
            logger.warning("python-i18n not installed, i18n disabled")
            return

        i18n.set('locale', default_locale)
        i18n.set('fallback', 'en')
        i18n.load_path.append(Path(locales_dir))
        i18n.set('filename_format', '{locale}.{format}')
        i18n.set('file_format', 'yaml')

        cls._initialized = True
        logger.info(f"i18n initialized: locales_dir={locales_dir}, default={default_locale}")

    @staticmethod
    def t(
        key: str,
        locale: Optional[str] = None,
        **kwargs: Any
    ) -> str:
        """
        Translate message key to user's language.

        Args:
            key: Translation key (e.g., 'commands.start.welcome')
            locale: Language code (defaults to current locale)
            **kwargs: Format variables for string interpolation

        Returns:
            Translated and formatted message

        Example:
            >>> I18nManager.t('errors.validation.empty_request', locale='pt')
            'Texto de solicitação vazio'

            >>> I18nManager.t('commands.start.version', locale='en', version='2.0')
            '🔧 Version: 2.0'
        """
        if i18n is None:
            # Fallback if i18n not available
            logger.warning(f"i18n not available, returning key: {key}")
            return key

        try:
            return i18n.t(key, locale=locale, **kwargs)
        except Exception as e:
            logger.error(f"Translation failed for key '{key}': {e}")
            return key

    @staticmethod
    def set_locale(locale: str):
        """
        Set current locale for thread.

        Args:
            locale: Language code (e.g., 'en', 'pt')
        """
        if i18n is None:
            return

        i18n.set('locale', locale)


# Convenience function for handlers
def t(key: str, session=None, **kwargs) -> str:
    """
    Shorthand translation function with session context.

    Args:
        key: Translation key
        session: User session (extracts language preference)
        **kwargs: Format variables

    Returns:
        Translated message in user's language
    """
    locale = session.language if session and hasattr(session, 'language') else None
    return I18nManager.t(key, locale=locale, **kwargs)
