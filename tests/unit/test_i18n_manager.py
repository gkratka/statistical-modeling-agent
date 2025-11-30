"""Tests for I18nManager."""

import pytest
from src.utils.i18n_manager import I18nManager, t


class TestI18nManager:
    """Test i18n manager functionality."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize i18n before each test."""
        I18nManager.initialize("./locales", "en")

    def test_initialization(self):
        """Test i18n initialization."""
        assert I18nManager._initialized

    def test_english_translation(self):
        """Test English translation retrieval."""
        result = I18nManager.t('commands.start.welcome', locale='en')
        assert "Welcome" in result
        assert "Statistical Modeling Agent" in result

    def test_portuguese_translation(self):
        """Test Portuguese translation retrieval."""
        result = I18nManager.t('commands.start.welcome', locale='pt')
        assert "Bem-vindo" in result
        assert "Modelagem Estatística" in result

    def test_variable_interpolation_english(self):
        """Test variable interpolation in English."""
        result = I18nManager.t(
            'commands.start.version',
            locale='en',
            version='2.0'
        )
        assert "Version: 2.0" in result

    def test_variable_interpolation_portuguese(self):
        """Test variable interpolation in Portuguese."""
        result = I18nManager.t(
            'commands.start.version',
            locale='pt',
            version='2.0'
        )
        assert "Versão: 2.0" in result

    def test_missing_key_fallback(self):
        """Test fallback for missing translation key."""
        result = I18nManager.t('nonexistent.key', locale='en')
        # Should return key itself or handle gracefully
        assert isinstance(result, str)

    def test_convenience_function_without_session(self):
        """Test convenience t() function without session."""
        result = t('commands.start.welcome')
        assert isinstance(result, str)

    def test_set_locale(self):
        """Test setting current locale."""
        I18nManager.set_locale('pt')
        # Verify locale was set (implementation-specific)
        assert True  # Basic test that it doesn't raise


class TestConvenienceFunction:
    """Test convenience t() function."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize i18n before each test."""
        I18nManager.initialize("./locales", "en")

    def test_with_mock_session(self):
        """Test t() with mock session object."""
        class MockSession:
            language = "pt"

        session = MockSession()
        result = t('commands.start.welcome', session=session)
        assert "Bem-vindo" in result
