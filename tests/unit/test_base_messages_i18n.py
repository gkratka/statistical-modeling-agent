"""Tests for base_messages.py i18n migration."""

import pytest
from src.bot.messages.base_messages import BaseMessages
from src.utils.i18n_manager import I18nManager


class TestBaseMessagesI18n:
    """Test i18n for BaseMessages formatting utilities."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Initialize i18n before each test."""
        I18nManager._initialized = False  # Reset initialization
        I18nManager.initialize("./locales", "en")

    def test_format_error_english(self):
        """Test error formatting in English."""
        result = BaseMessages.format_error(
            title="Validation Error",
            details="The file could not be validated.",
            suggestions=["Check file format", "Try again"],
            locale="en"
        )

        assert "❌" in result
        assert "Validation Error" in result
        assert "The file could not be validated" in result
        assert "💡 Suggestions" in result
        assert "Check file format" in result

    def test_format_error_portuguese(self):
        """Test error formatting in Portuguese."""
        result = BaseMessages.format_error(
            title="Erro de Validação",
            details="O arquivo não pôde ser validado.",
            suggestions=["Verificar formato do arquivo", "Tentar novamente"],
            locale="pt"
        )

        assert "❌" in result
        assert "Erro de Validação" in result
        assert "O arquivo não pôde ser validado" in result
        assert "💡 Sugestões" in result
        assert "Verificar formato do arquivo" in result

    def test_format_error_no_suggestions_english(self):
        """Test error formatting without suggestions in English."""
        result = BaseMessages.format_error(
            title="Error",
            details="Something went wrong.",
            locale="en"
        )

        assert "❌" in result
        assert "Error" in result
        assert "Something went wrong" in result
        # Should not have suggestions section
        assert "Suggestions" not in result

    def test_format_success_english(self):
        """Test success formatting in English."""
        result = BaseMessages.format_success(
            title="Model Trained",
            summary={"Model ID": "rf_123", "Accuracy": "95%"},
            locale="en"
        )

        assert "✅" in result
        assert "Model Trained" in result
        assert "📊 Summary" in result
        assert "Model ID" in result
        assert "rf_123" in result

    def test_format_success_portuguese(self):
        """Test success formatting in Portuguese."""
        result = BaseMessages.format_success(
            title="Modelo Treinado",
            summary={"ID do Modelo": "rf_123", "Precisão": "95%"},
            locale="pt"
        )

        assert "✅" in result
        assert "Modelo Treinado" in result
        assert "📊 Resumo" in result
        assert "ID do Modelo" in result
        assert "rf_123" in result

    def test_format_success_with_metrics_english(self):
        """Test success formatting with metrics in English."""
        result = BaseMessages.format_success(
            title="Training Complete",
            summary={"Duration": "5 minutes"},
            metrics={"Accuracy": 0.95, "Loss": 0.05},
            locale="en"
        )

        assert "Training Complete" in result
        assert "📊 Summary" in result
        assert "📈 Metrics" in result
        assert "Accuracy: 0.9500" in result
        assert "Loss: 0.0500" in result

    def test_format_success_with_next_steps_english(self):
        """Test success formatting with next steps in English."""
        result = BaseMessages.format_success(
            title="Data Loaded",
            summary={"Rows": "1000"},
            next_steps=["Select target column", "Choose model type"],
            locale="en"
        )

        assert "Data Loaded" in result
        assert "💡 Next Steps" in result
        assert "Select target column" in result
        assert "Choose model type" in result

    def test_format_success_with_next_steps_portuguese(self):
        """Test success formatting with next steps in Portuguese."""
        result = BaseMessages.format_success(
            title="Dados Carregados",
            summary={"Linhas": "1000"},
            next_steps=["Selecionar coluna alvo", "Escolher tipo de modelo"],
            locale="pt"
        )

        assert "Dados Carregados" in result
        assert "💡 Próximos Passos" in result
        assert "Selecionar coluna alvo" in result

    def test_format_list_no_truncation(self):
        """Test list formatting without truncation."""
        result = BaseMessages.format_list(
            items=["item1", "item2", "item3"],
            max_items=10,
            locale="en"
        )

        assert "• item1" in result
        assert "• item2" in result
        assert "• item3" in result
        assert "more" not in result

    def test_format_list_with_truncation_english(self):
        """Test list formatting with truncation in English."""
        items = [f"item{i}" for i in range(15)]
        result = BaseMessages.format_list(
            items=items,
            max_items=10,
            locale="en"
        )

        assert "• item0" in result
        assert "• item9" in result
        assert "... and 5 more" in result
        assert "item10" not in result

    def test_format_list_with_truncation_portuguese(self):
        """Test list formatting with truncation in Portuguese."""
        items = [f"item{i}" for i in range(15)]
        result = BaseMessages.format_list(
            items=items,
            max_items=10,
            locale="pt"
        )

        assert "• item0" in result
        assert "• item9" in result
        assert "... e mais 5" in result
        assert "item10" not in result

    def test_format_progress_english(self):
        """Test progress formatting in English."""
        result = BaseMessages.format_progress(
            phase="Training Model",
            current=2,
            total=5,
            elapsed_seconds=10.5,
            locale="en"
        )

        assert "🔄" in result
        assert "Phase 2/5" in result
        assert "Training Model" in result
        assert "⏱️ Elapsed: 10.5s" in result

    def test_format_progress_portuguese(self):
        """Test progress formatting in Portuguese."""
        result = BaseMessages.format_progress(
            phase="Treinando Modelo",
            current=2,
            total=5,
            elapsed_seconds=10.5,
            locale="pt"
        )

        assert "🔄" in result
        assert "Fase 2/5" in result
        assert "Treinando Modelo" in result
        assert "⏱️ Decorrido: 10.5s" in result

    def test_format_progress_no_elapsed_time(self):
        """Test progress formatting without elapsed time."""
        result = BaseMessages.format_progress(
            phase="Loading Data",
            current=1,
            total=3,
            locale="en"
        )

        assert "Phase 1/3" in result
        assert "Loading Data" in result
        assert "Elapsed" not in result

    def test_format_metric_value_float(self):
        """Test metric value formatting for floats."""
        result = BaseMessages._format_metric_value(0.123456)
        assert result == "0.1235"

    def test_format_metric_value_int(self):
        """Test metric value formatting for integers."""
        result = BaseMessages._format_metric_value(1000000)
        assert result == "1,000,000"

    def test_format_metric_value_string(self):
        """Test metric value formatting for strings."""
        result = BaseMessages._format_metric_value("custom value")
        assert result == "custom value"

    def test_format_error_default_locale(self):
        """Test error formatting with default locale (English)."""
        result = BaseMessages.format_error(
            title="Error",
            details="Details"
        )

        # Should default to English
        assert "❌" in result
        assert "💡 Suggestions" not in result  # No suggestions provided

    def test_format_success_default_locale(self):
        """Test success formatting with default locale (English)."""
        result = BaseMessages.format_success(
            title="Success",
            summary={"Key": "Value"}
        )

        # Should default to English
        assert "✅" in result
        assert "📊 Summary" in result
