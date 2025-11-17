"""Tests for handlers.py i18n integration."""

import pytest
from src.utils.i18n_manager import I18nManager


class TestHandlersI18n:
    """Test i18n integration for handlers.py commands."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize i18n before each test."""
        I18nManager.initialize("./locales", "en")

    # =========================================================================
    # /start Command
    # =========================================================================

    def test_start_welcome_english(self):
        """Test /start welcome message in English."""
        result = I18nManager.t('commands.start.welcome', locale='en')
        assert "Welcome to the Statistical Modeling Agent" in result

    def test_start_welcome_portuguese(self):
        """Test /start welcome message in Portuguese."""
        result = I18nManager.t('commands.start.welcome', locale='pt')
        assert "Bem-vindo ao Agente de Modelagem Estatística" in result

    def test_start_version_english(self):
        """Test /start version display in English."""
        result = I18nManager.t('commands.start.version', locale='en', version='v2.0')
        assert "Version:" in result
        assert "v2.0" in result

    def test_start_version_portuguese(self):
        """Test /start version display in Portuguese."""
        result = I18nManager.t('commands.start.version', locale='pt', version='v2.0')
        assert "Versão:" in result
        assert "v2.0" in result

    def test_start_features_english(self):
        """Test /start features list in English."""
        result = I18nManager.t('commands.start.features', locale='en')
        assert "Statistical analysis" in result
        assert "Machine learning" in result
        assert "predictions and insights" in result

    def test_start_features_portuguese(self):
        """Test /start features list in Portuguese."""
        result = I18nManager.t('commands.start.features', locale='pt')
        assert "Análise estatística" in result
        assert "machine learning" in result

    def test_start_instructions_english(self):
        """Test /start instructions in English."""
        result = I18nManager.t('commands.start.instructions', locale='en')
        assert "Getting Started" in result
        assert "/help" in result
        assert "/train" in result

    def test_start_instructions_portuguese(self):
        """Test /start instructions in Portuguese."""
        result = I18nManager.t('commands.start.instructions', locale='pt')
        assert "Primeiros Passos" in result
        assert "/help" in result

    # =========================================================================
    # /help Command
    # =========================================================================

    def test_help_title_english(self):
        """Test /help title in English."""
        result = I18nManager.t('commands.help.title', locale='en')
        assert "Statistical Modeling Agent Help" in result

    def test_help_title_portuguese(self):
        """Test /help title in Portuguese."""
        result = I18nManager.t('commands.help.title', locale='pt')
        assert "Ajuda do Agente de Modelagem Estatística" in result

    def test_help_description_english(self):
        """Test /help description in English."""
        result = I18nManager.t('commands.help.description', locale='en')
        assert "AI assistant" in result
        assert "statistical analysis" in result

    def test_help_description_portuguese(self):
        """Test /help description in Portuguese."""
        result = I18nManager.t('commands.help.description', locale='pt')
        assert "assistente de IA" in result

    def test_help_commands_section_english(self):
        """Test /help commands section header in English."""
        result = I18nManager.t('commands.help.commands_section', locale='en')
        assert "Available Commands" in result

    def test_help_commands_section_portuguese(self):
        """Test /help commands section header in Portuguese."""
        result = I18nManager.t('commands.help.commands_section', locale='pt')
        assert "Comandos Disponíveis" in result

    def test_help_start_cmd_english(self):
        """Test /start command description in English."""
        result = I18nManager.t('commands.help.start_cmd', locale='en')
        assert "/start" in result
        assert "Start the bot" in result

    def test_help_start_cmd_portuguese(self):
        """Test /start command description in Portuguese."""
        result = I18nManager.t('commands.help.start_cmd', locale='pt')
        assert "/start" in result
        assert "Iniciar o bot" in result

    def test_help_help_cmd_english(self):
        """Test /help command description in English."""
        result = I18nManager.t('commands.help.help_cmd', locale='en')
        assert "/help" in result
        assert "help message" in result

    def test_help_train_cmd_english(self):
        """Test /train command description in English."""
        result = I18nManager.t('commands.help.train_cmd', locale='en')
        assert "/train" in result
        assert "ML model training" in result

    def test_help_train_cmd_portuguese(self):
        """Test /train command description in Portuguese."""
        result = I18nManager.t('commands.help.train_cmd', locale='pt')
        assert "/train" in result
        assert "treinamento" in result

    def test_help_models_cmd_english(self):
        """Test /models command description in English."""
        result = I18nManager.t('commands.help.models_cmd', locale='en')
        assert "/models" in result
        assert "trained models" in result

    def test_help_models_cmd_portuguese(self):
        """Test /models command description in Portuguese."""
        result = I18nManager.t('commands.help.models_cmd', locale='pt')
        assert "/models" in result
        assert "modelos treinados" in result

    def test_help_predict_cmd_english(self):
        """Test /predict command description in English."""
        result = I18nManager.t('commands.help.predict_cmd', locale='en')
        assert "/predict" in result
        assert "predictions" in result

    def test_help_predict_cmd_portuguese(self):
        """Test /predict command description in Portuguese."""
        result = I18nManager.t('commands.help.predict_cmd', locale='pt')
        assert "/predict" in result
        assert "previsões" in result

    def test_help_features_section_english(self):
        """Test /help features section in English."""
        result = I18nManager.t('commands.help.features_section', locale='en')
        assert "Features:" in result
        assert "Statistical Analysis" in result
        assert "ML Training" in result
        assert "Predictions" in result

    def test_help_features_section_portuguese(self):
        """Test /help features section in Portuguese."""
        result = I18nManager.t('commands.help.features_section', locale='pt')
        assert "Recursos:" in result
        assert "Análise Estatística" in result

    def test_help_support_english(self):
        """Test /help support message in English."""
        result = I18nManager.t('commands.help.support', locale='en')
        assert "Need help" in result
        assert "natural language" in result

    def test_help_support_portuguese(self):
        """Test /help support message in Portuguese."""
        result = I18nManager.t('commands.help.support', locale='pt')
        assert "Precisa de ajuda" in result
        assert "linguagem natural" in result
